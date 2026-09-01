"""
Deployment-realistic evaluation: run the streaming EMA detector over each VAL clip
(same split as train.py) and score clip-level detection. Sweeps EMA threshold and
also reports the raw max-window-prob metric for comparison.
"""
import os, numpy as np, torch, torchvision, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (average_precision_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
from dataset import make_window_feat, load_manifest
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
CKPT  = os.path.join(HERE, "runs", "best.pt")

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(CKPT, map_location=dev)
W = ck['args']['W']
m = torchvision.models.mobilenet_v3_small(); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
m.load_state_dict(ck['model']); m.to(dev).eval()

man = load_manifest()
_, va = train_test_split(man, test_size=0.2, stratify=[r['label'] for r in man], random_state=42)

@torch.no_grad()
def clip_scores(rec, stride=4, alpha=0.4):
    stack = np.load(os.path.join(CACHE, rec['cls'], rec['clip'] + '.npy'))
    starts = range(0, max(1, len(stack) - W + 1), stride)
    ema = 0.0; maxp = 0.0; maxema = 0.0
    for s in starts:
        feat = torch.from_numpy(make_window_feat(stack, s, W))[None].to(dev)
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
    print(f"\n[{name}] AP={ap:.3f}  bestF1={best[0]:.3f} @thr={best[1]:.2f}  P={best[2]:.3f} R={best[3]:.3f}")

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
sweep(maxema, "max-EMA (streaming/deployment rule)")
sweep_recall_first(maxema, "max-EMA (deployment)")

# report block + curves at the deployment operating point: zero-miss if reachable, else best-F2
op_thr = zero_miss_thr(maxema)
op_name = "max-EMA @ zero-miss (recall-first)"
if op_thr is None:
    op_thr, op_name = best_fbeta_thr(maxema), "max-EMA @ best-F2 (no zero-miss point on VAL)"
report_metrics(maxema, op_name, op_thr)
save_curves(maxema, "max-EMA", op_thr)
