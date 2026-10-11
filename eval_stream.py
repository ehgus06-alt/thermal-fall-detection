"""
Deployment-realistic evaluation: run the streaming EMA detector over each VAL clip
(same split as train.py) and score clip-level detection. Sweeps EMA threshold and
also reports the raw max-window-prob metric for comparison.
"""
import os, argparse, numpy as np, torch, torchvision, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from dataset import make_window_feat, load_manifest

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

_ap = argparse.ArgumentParser()
_ap.add_argument('--ckpt', default=os.path.join(HERE, "runs", "best.pt"),
                 help='checkpoint to evaluate (default runs/best.pt)')
_ap.add_argument('--split_seed', type=int, default=42,
                 help='train/val split seed; MUST match train.py --split_seed so the same val set is scored')
_ap.add_argument('--json_out', default='', help='if set, also write the REPORT METRICS dict here as JSON')
_ap.add_argument('--wandb', action='store_true', help='log deployment metrics + curves to Weights & Biases')
_ap.add_argument('--wandb_project', default='thermal-fall')
_ap.add_argument('--wandb_run', default='eval-stream')
ARGS = _ap.parse_args()

CKPT = ARGS.ckpt

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(CKPT, map_location=dev)
W = ck['args']['W']
MODE = ck['args'].get('input_mode', 'full')   # window features must match how the model was trained
m = torchvision.models.mobilenet_v3_small(); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
m.load_state_dict(ck['model']); m.to(dev).eval()
print(f"eval ckpt={os.path.basename(CKPT)}  input_mode={MODE}  W={W}  split_seed={ARGS.split_seed}", flush=True)

man = load_manifest()
_, va = train_test_split(man, test_size=0.2, stratify=[r['label'] for r in man],
                         random_state=ARGS.split_seed)

@torch.no_grad()
def clip_scores(rec, stride=4, alpha=0.4):
    stack = np.load(os.path.join(CACHE, rec['cls'], rec['clip'] + '.npy'))
    starts = range(0, max(1, len(stack) - W + 1), stride)
    ema = 0.0; maxp = 0.0; maxema = 0.0
    for s in starts:
        feat = torch.from_numpy(make_window_feat(stack, s, W, input_mode=MODE))[None].to(dev)
        with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
            p = torch.sigmoid(m(feat).squeeze()).item()
        ema = alpha * p + (1 - alpha) * ema
        maxp = max(maxp, p); maxema = max(maxema, ema)
    return maxp, maxema

y = np.array([r['label'] for r in va])
maxp = np.zeros(len(va)); maxema = np.zeros(len(va))
for i, rec in enumerate(va):
    maxp[i], maxema[i] = clip_scores(rec)
    print(f"{rec['cls']}/{rec['clip']:>10} y={rec['label']} maxp={maxp[i]:.2f} maxEMA={maxema[i]:.2f}", flush=True)

def sweep(score, name):
    ap = average_precision_score(y, score)
    best = (0, 0)
    for t in np.linspace(0.05, 0.95, 37):
        pr, rc, f1, _ = precision_recall_fscore_support(y, (score >= t).astype(int), average='binary', zero_division=0)
        if f1 > best[0]: best = (f1, t, pr, rc)
    # confusion matrix + specificity at the SAME best-F1 threshold (so P/R/F1/Spec are one operating point)
    tn, fp, fn, tp = confusion_matrix(y, (score >= best[1]).astype(int), labels=[0, 1]).ravel()
    spec = tn / max(tn + fp, 1)
    print(f"\n[{name}] AP={ap:.3f}  bestF1={best[0]:.3f} @thr={best[1]:.2f}  "
          f"P={best[2]:.3f} R={best[3]:.3f} Spec={spec:.3f}")
    print(f"         confusion @bestF1: TP={tp} FP={fp} FN={fn} TN={tn}  (missed falls={fn})")
    return dict(ap=float(ap), f1=float(best[0]), thr=float(best[1]), precision=float(best[2]),
                recall=float(best[3]), specificity=float(spec),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn))

def sweep_recall_first(score, name, beta=2.0):
    """RECALL-FIRST operating point: a missed fall is far worse than a false alarm
    (caregiver re-checks a false alarm; a missed fall is dangerous).
    Reports (1) the HIGHEST threshold that still catches EVERY fall on VAL -> the safest
    zero-miss point with the fewest false alarms, and (2) the best F-beta (recall weighted
    beta x). Lower threshold = higher recall, so the zero-miss point sits at a low thr."""
    thrs = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thrs:
        pred = (score >= t).astype(int)
        pr, rc, _, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        fb = (1 + beta**2) * pr * rc / max(beta**2 * pr + rc, 1e-9)
        rows.append((t, pr, rc, fb))
    print(f"\n[{name}] RECALL-FIRST (missed fall >> false alarm)")
    zero_miss = [r for r in rows if r[2] >= 0.999]
    if zero_miss:
        t, pr, rc, _ = max(zero_miss, key=lambda r: r[0])   # highest thr that still misses nothing
        print(f"  zero-miss point:  --thr {t:.2f}  ->  catches ALL falls (R=1.00), "
              f"P={pr:.2f}  ({1-pr:.0%} of alerts are false = caregiver re-checks)")
    else:
        print("  !! no threshold reaches R=1.00 on VAL -> the MODEL itself misses some falls; "
              "lower thr won't help. Enable --collapse-trigger as a second net and/or retrain.")
    t, pr, rc, fb = max(rows, key=lambda r: r[3])
    print(f"  best F{beta:.0f} (recall-weighted): --thr {t:.2f}  P={pr:.2f} R={rc:.2f}")
    print("  --- full curve (pick your own comfort level) ---")
    print("   thr :  P  /  R   (missed falls)")
    for t in np.linspace(0.20, 0.70, 11):
        pred = (score >= t).astype(int)
        pr, rc, _, _ = precision_recall_fscore_support(y, pred, average='binary', zero_division=0)
        missed = int(((y == 1) & (pred == 0)).sum())
        print(f"  {t:.2f} : {pr:.2f} / {rc:.2f}   ({missed} missed)")

def zero_miss_thr(score):
    """Highest threshold that still catches EVERY fall (recall=1.0), or None if unreachable."""
    best = None
    for t in np.linspace(0.05, 0.95, 91):
        if (score[y == 1] >= t).mean() >= 0.999:   # recall on the fall clips only
            best = t
    return best

def best_fbeta_thr(score, beta=2.0):
    """Threshold maximizing F-beta (beta>1 favors recall)."""
    best = (0.0, 0.5)
    for t in np.linspace(0.05, 0.95, 91):
        pr, rc, _, _ = precision_recall_fscore_support(y, (score >= t).astype(int),
                                                       average='binary', zero_division=0)
        fb = (1 + beta**2) * pr * rc / max(beta**2 * pr + rc, 1e-9)
        if fb > best[0]: best = (fb, t)
    return best[1]

def report_metrics(score, name, thr):
    """Report-ready block at ONE operating threshold: confusion matrix + every derived rate,
    plus a copy-paste CSV row for the report table."""
    pred = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    recall      = tp / max(tp + fn, 1)     # sensitivity - the safety-critical metric
    precision   = tp / max(tp + fp, 1)
    specificity = tn / max(tn + fp, 1)
    far         = fp / max(fp + tn, 1)     # false-alarm rate
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-9)
    auroc = roc_auc_score(y, score) if len(set(y.tolist())) > 1 else float('nan')
    ap    = average_precision_score(y, score)
    print(f"\n========== REPORT METRICS [{name}] @thr={thr:.2f} ==========")
    print(f"  confusion matrix     TP={tp}  FP={fp}  FN={fn}  TN={tn}   (N={len(y)}, falls={int(y.sum())})")
    print(f"  Recall / Sensitivity {recall:.3f}   <- missed falls: {fn}")
    print(f"  Precision            {precision:.3f}")
    print(f"  Specificity          {specificity:.3f}")
    print(f"  False-alarm rate     {far:.3f}")
    print(f"  F1 / F2              {f1:.3f} / {f2:.3f}   (F2 weights recall 2x)")
    print(f"  AUROC / AP(AUPRC)    {auroc:.3f} / {ap:.3f}   (threshold-independent)")
    print(f"  --- copy-paste CSV row ---")
    print(f"  Recall,Precision,Specificity,F1,F2,AUROC,AP,MissedFalls")
    print(f"  {recall:.3f},{precision:.3f},{specificity:.3f},{f1:.3f},{f2:.3f},{auroc:.3f},{ap:.3f},{fn}")
    return dict(recall=recall, precision=precision, specificity=specificity, far=far,
                f1=f1, f2=f2, auroc=auroc, ap=ap, thr=float(thr),
                TP=int(tp), FP=int(fp), FN=int(fn), TN=int(tn), missed=int(fn))

def save_curves(score, name, thr_mark=None, out_dir=os.path.join(HERE, 'runs')):
    """Save ROC + PR curves as one PNG for the report. The chosen operating point (thr_mark)
    is drawn as a red dot on both. Skips gracefully if matplotlib isn't installed."""
    try:
        import matplotlib
        matplotlib.use('Agg')                 # headless: no display needed
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[curves] matplotlib not available ({e}) - skipping PNG (pip install matplotlib)")
        return
    fpr, tpr, _ = roc_curve(y, score)
    prec, rec, _ = precision_recall_curve(y, score)
    auroc = roc_auc_score(y, score); ap = average_precision_score(y, score)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    ax[0].plot(fpr, tpr, lw=2, label=f'AUROC={auroc:.3f}')
    ax[0].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    ax[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate (Recall)',
              title=f'ROC - {name}', xlim=(0, 1), ylim=(0, 1.02))
    base = float(y.mean())
    ax[1].plot(rec, prec, lw=2, label=f'AP={ap:.3f}')
    ax[1].axhline(base, ls='--', color='gray', lw=1, label=f'baseline={base:.2f}')
    ax[1].set(xlabel='Recall', ylabel='Precision', title=f'PR - {name}', xlim=(0, 1), ylim=(0, 1.02))
    if thr_mark is not None:                  # mark the deployment operating point
        pred = (score >= thr_mark).astype(int)
        tp = ((y == 1) & (pred == 1)).sum(); fp = ((y == 0) & (pred == 1)).sum()
        fn = ((y == 1) & (pred == 0)).sum(); tn = ((y == 0) & (pred == 0)).sum()
        op_tpr = tp / max(tp + fn, 1); op_fpr = fp / max(fp + tn, 1); op_prec = tp / max(tp + fp, 1)
        ax[0].scatter([op_fpr], [op_tpr], color='red', zorder=5, label=f'op thr={thr_mark:.2f}')
        ax[1].scatter([op_tpr], [op_prec], color='red', zorder=5, label=f'op thr={thr_mark:.2f}')
    ax[0].legend(loc='lower right'); ax[1].legend(loc='lower left')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'roc_pr.png')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print(f"[curves] saved -> {path}")

print("\n===== VAL streaming eval (n=%d, pos=%d) =====" % (len(y), int(y.sum())))
sweep(maxp,  "max-window-prob (matches training)")
bestf1 = sweep(maxema, "max-EMA (streaming/deployment rule)")   # best-F1 operating point (paper-quoted)
sweep_recall_first(maxema, "max-EMA (deployment)")
bestf1["auroc"] = float(roc_auc_score(y, maxema)) if len(set(y.tolist())) > 1 else float('nan')

# report block + curves at the deployment operating point: zero-miss if reachable, else best-F2
op_thr = zero_miss_thr(maxema)
op_name = "max-EMA @ zero-miss (recall-first)"
if op_thr is None:
    op_thr, op_name = best_fbeta_thr(maxema), "max-EMA @ best-F2 (no zero-miss point on VAL)"
rep = report_metrics(maxema, op_name, op_thr)
save_curves(maxema, "max-EMA", op_thr)

if ARGS.json_out:
    import json
    # top level = best-F1 operating point (AP/AUROC are threshold-free; F1/P/R/Spec/confusion @bestF1),
    # so the ablation table's streaming row is ONE consistent operating point.
    out = dict(bestf1)
    out.update(input_mode=MODE, ckpt=os.path.basename(CKPT), split_seed=ARGS.split_seed,
               n_val=len(y), n_pos=int(y.sum()), operating_point='best-F1',
               recall_first=dict(rep, operating_point=op_name))   # zero-miss/F2 deployment point (nested)
    with open(ARGS.json_out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"[json] wrote streaming metrics -> {ARGS.json_out}", flush=True)

if ARGS.wandb:
    import wandb
    run = wandb.init(project=ARGS.wandb_project, name=ARGS.wandb_run,
                     config=dict(operating_point=op_name, W=int(W), n_val=len(y), n_pos=int(y.sum())))
    run.summary.update({f'deploy/{k}': v for k, v in rep.items()})   # recall/precision/f1/f2/auroc/ap/...
    # PR + ROC curves as native W&B plots (interactive) + the saved PNG
    probs2 = np.stack([1 - maxema, maxema], axis=1)
    run.log({'pr_curve':  wandb.plot.pr_curve(y, probs2, labels=['nonfall', 'fall']),
             'roc_curve': wandb.plot.roc_curve(y, probs2, labels=['nonfall', 'fall']),
             'confusion': wandb.plot.confusion_matrix(y_true=y.tolist(),
                          preds=(maxema >= op_thr).astype(int).tolist(), class_names=['nonfall', 'fall'])})
    png = os.path.join(HERE, 'runs', 'roc_pr.png')
    if os.path.exists(png):
        run.log({'roc_pr_png': wandb.Image(png)})
    run.finish()
    print(f"[wandb] logged deployment metrics -> project '{ARGS.wandb_project}' run '{ARGS.wandb_run}'")
