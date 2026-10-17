"""
Temporal-robustness evaluation (no retraining): re-score the ablation checkpoints on the SAME
val split after perturbing the clips along the time axis, to test whether the MHI summary
(full) degrades less than the raw frame stack (stack) when the frame rate changes or frames drop.

Conditions (applied to the cached Lepton-emulated clip before windowing):
  orig    unchanged                                   (reference; must reproduce stream_<tag>.json)
  half    every 2nd frame      (x0.5 fps)
  third   every 3rd frame      (x0.33 fps)
  drop20  20% of frames removed at random (deterministic per clip -> identical for every model)
  double  every frame shown twice (x2 fps, motion looks slower)

Scoring is the deployment rule of eval_stream.py (stride 4 windows, EMA a=0.4, max over time,
best-F1 over 37 thresholds). Also reported: recall/specificity at the threshold that was best on
`orig` for that checkpoint -> what happens if the alarm threshold is calibrated on clean data and
the camera then runs at a different rate.

  python robust_eval.py                                  # stack+full, seeds 0-2, all conditions
  python robust_eval.py --modes still stack no_mhi full  # every ablation mode
Writes runs/robust_results.json and runs/robust_table.md (per-checkpoint caches in runs/robust/).
"""
import os, json, zlib, argparse, numpy as np, torch, torchvision, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from dataset import make_window_feat, load_manifest

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
RUNS  = os.path.join(HERE, "runs")
CONDS = ['orig', 'half', 'third', 'drop20', 'double']
COND_DESC = {'orig': 'unchanged', 'half': 'x0.5 fps (every 2nd frame)', 'third': 'x0.33 fps (every 3rd frame)',
             'drop20': '20% random frame drops', 'double': 'x2 fps (each frame repeated)'}

def perturb(st, cond, clip):
    if cond == 'orig':   return st
    if cond == 'half':   return st[::2]
    if cond == 'third':  return st[::3]
    if cond == 'double': return np.repeat(st, 2, axis=0)
    if cond == 'drop20':
        rng = np.random.default_rng(zlib.crc32(clip.encode()))   # same drops for every model
        keep = rng.random(len(st)) >= 0.2
        keep[0] = keep[-1] = True
        return st[keep]
    raise ValueError(cond)

@torch.no_grad()
def clip_score(model, st, W, mode, dev, stride=4, alpha=0.4, bs=64):
    starts = list(range(0, max(1, len(st) - W + 1), stride))
    probs = []
    for i in range(0, len(starts), bs):
        x = torch.from_numpy(np.stack([make_window_feat(st, s, W, input_mode=mode)
                                       for s in starts[i:i + bs]])).to(dev)
        with torch.amp.autocast('cuda', enabled=(dev == 'cuda')):
            probs.append(torch.sigmoid(model(x).float().squeeze(1)).cpu().numpy())
    ema = mx = 0.0
    for p in np.concatenate(probs):
        ema = alpha * float(p) + (1 - alpha) * ema; mx = max(mx, ema)
    return mx

def at_thr(y, sc, t):
    pr = (sc >= t).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum())
    fn = int(((pr == 0) & (y == 1)).sum()); tn = int(((pr == 0) & (y == 0)).sum())
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(f1=2 * p * r / max(p + r, 1e-9), precision=p, recall=r, specificity=tn / max(tn + fp, 1),
                TP=tp, FP=fp, FN=fn, TN=tn)

def metrics(y, sc, ref_thr=None):
    best = max(((at_thr(y, sc, t), float(t)) for t in np.linspace(0.05, 0.95, 37)), key=lambda b: b[0]['f1'])
    out = dict(ap=float(average_precision_score(y, sc)), auroc=float(roc_auc_score(y, sc)), thr=best[1],
               **{k: float(v) for k, v in best[0].items()})
    if ref_thr is not None:
        f = at_thr(y, sc, ref_thr)
        out['at_orig_thr'] = dict(thr=ref_thr, recall=f['recall'], specificity=f['specificity'], f1=f['f1'])
    return out

def ms(v):
    v = np.asarray(v, float); return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0, n=len(v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=['stack', 'full'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--conds', nargs='+', default=CONDS, choices=CONDS)
    ap.add_argument('--split_seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='debug: only the first N val clips (no caching)')
    args = ap.parse_args()
    if 'orig' not in args.conds: args.conds = ['orig'] + args.conds
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    man = load_manifest()
    _, va = train_test_split(man, test_size=0.2, stratify=[r['label'] for r in man], random_state=args.split_seed)
    if args.limit: va = va[:args.limit]
    y = np.array([r['label'] for r in va])
    print(f"val clips: {len(va)} (fall {int(y.sum())} / ADL {int((1 - y).sum())})", flush=True)
    os.makedirs(os.path.join(RUNS, 'robust'), exist_ok=True)

    res = {}                                            # res[mode][seed][cond] = metrics
    for mode in args.modes:
        for seed in args.seeds:
            tag = f"_{mode}_s{seed}"
            cache = os.path.join(RUNS, 'robust', f"robust{tag}.json")
            ckpt = os.path.join(RUNS, f"best{tag}.pt")
            if not os.path.exists(ckpt):
                print(f"!! missing {ckpt}, skipped", flush=True); continue
            done = {}
            if not args.limit and os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(ckpt):
                done = json.load(open(cache, encoding='utf-8'))
            ck = torch.load(ckpt, map_location=dev, weights_only=False)
            W, cmode = ck['args']['W'], ck['args'].get('input_mode', 'full')
            assert cmode == mode, (ckpt, cmode)
            model = torchvision.models.mobilenet_v3_small()
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
            model.load_state_dict(ck['model']); model.to(dev).eval()
            for cond in args.conds:
                if cond in done: continue
                sc = np.array([clip_score(model, perturb(np.load(os.path.join(CACHE, r['cls'], r['clip'] + '.npy')),
                                                         cond, r['clip']), W, mode, dev) for r in va])
                done[cond] = metrics(y, sc, None if cond == 'orig' else done['orig']['thr'])
                m = done[cond]
                print(f"{tag:<10} {cond:<7} AP={m['ap']:.3f} AUROC={m['auroc']:.3f} F1={m['f1']:.3f} "
                      f"Spec={m['specificity']:.3f}" + (f" | @orig-thr {m['at_orig_thr']['thr']:.2f}: "
                      f"R={m['at_orig_thr']['recall']:.3f} Spec={m['at_orig_thr']['specificity']:.3f}"
                      if 'at_orig_thr' in m else ""), flush=True)
                if not args.limit:
                    json.dump(done, open(cache, 'w', encoding='utf-8'), indent=1)
            ref = os.path.join(RUNS, f"stream{tag}.json")    # sanity: orig must match eval_stream
            if os.path.exists(ref) and not args.limit:
                d = done['orig']['ap'] - json.load(open(ref))['ap']
                print(f"   sanity {tag}: orig AP - eval_stream AP = {d:+.4f}", flush=True)
            res.setdefault(mode, {})[seed] = done

    # ---- aggregate: absolute metrics, drop vs orig (paired per checkpoint), full-vs-stack paired delta ----
    keys = ['ap', 'auroc', 'f1', 'specificity']
    summ = {}
    for mode, bys in res.items():
        summ[mode] = {}
        for cond in args.conds:
            seeds = [s for s in bys if cond in bys[s]]
            row = {k: ms([bys[s][cond][k] for s in seeds]) for k in keys}
            row['drop'] = {k: ms([bys[s][cond][k] - bys[s]['orig'][k] for s in seeds]) for k in keys}
            if cond != 'orig':
                row['at_orig_thr'] = {k: ms([bys[s][cond]['at_orig_thr'][k] for s in seeds])
                                      for k in ('recall', 'specificity')}
            summ[mode][cond] = row
    paired = {}
    if 'full' in res and 'stack' in res:
        for cond in args.conds:
            if cond == 'orig': continue
            seeds = [s for s in res['full'] if s in res['stack'] and cond in res['full'][s] and cond in res['stack'][s]]
            paired[cond] = {k: ms([(res['full'][s][cond][k] - res['full'][s]['orig'][k]) -
                                   (res['stack'][s][cond][k] - res['stack'][s]['orig'][k]) for s in seeds])
                            for k in keys}
    out = dict(val_clips=len(va), conditions=COND_DESC, seeds=args.seeds, per_checkpoint=res,
               summary=summ, full_minus_stack_drop=paired)
    sfx = '_debug' if args.limit else ''
    json.dump(out, open(os.path.join(RUNS, f'robust_results{sfx}.json'), 'w', encoding='utf-8'), indent=1)

    f = lambda d: f"{d['mean']:.3f} ± {d['std']:.3f}"
    g = lambda d: f"{d['mean']:+.3f} ± {d['std']:.3f}"
    L = ["# Temporal robustness (no retraining)", "",
         f"Val split seed {args.split_seed}: {len(va)} clips (fall {int(y.sum())} / ADL {int((1 - y).sum())}). "
         f"Checkpoints: ablation runs, seeds {args.seeds}. Streaming rule: stride 4, EMA 0.4, max. "
         "Mean ± std over seeds.", "",
         "Conditions: " + "; ".join(f"`{c}` = {COND_DESC[c]}" for c in args.conds), "",
         "## Streaming metrics per condition", "",
         "| mode | condition | AP | ΔAP vs orig | AUROC | best-F1 | Spec @best-F1 | R @orig-thr | Spec @orig-thr |",
         "|---|---|---|---|---|---|---|---|---|"]
    for mode in summ:
        for cond in args.conds:
            r = summ[mode][cond]
            ot = r.get('at_orig_thr')
            L.append(f"| {mode} | {cond} | {f(r['ap'])} | {g(r['drop']['ap']) if cond != 'orig' else '—'} | "
                     f"{f(r['auroc'])} | {f(r['f1'])} | {f(r['specificity'])} | "
                     f"{f(ot['recall']) if ot else '—'} | {f(ot['specificity']) if ot else '—'} |")
    if paired:
        L += ["", "## Is full (MHI) more robust than stack?", "",
              "Paired by seed: (full drop) − (stack drop). **Positive = full degrades less.** "
              "A claim needs the mean clearly beyond ±std with the same sign for every seed.", "",
              "| condition | ΔAP | ΔAUROC | ΔF1 | ΔSpec |", "|---|---|---|---|---|"]
        for cond, d in paired.items():
            L.append(f"| {cond} | {g(d['ap'])} | {g(d['auroc'])} | {g(d['f1'])} | {g(d['specificity'])} |")
    L += ["", "R/Spec @orig-thr: alarm threshold fixed at the best-F1 threshold found on `orig` for that "
          "checkpoint (calibrate on clean data, then the camera rate changes)."]
    open(os.path.join(RUNS, f'robust_table{sfx}.md'), 'w', encoding='utf-8').write("\n".join(L) + "\n")
    print("\n".join(L))

if __name__ == '__main__':
    main()
