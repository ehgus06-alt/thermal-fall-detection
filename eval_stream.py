"""
Deployment-realistic evaluation: run the streaming EMA detector over each VAL clip
(same split as train.py) and score clip-level detection. Sweeps EMA threshold and
also reports the raw max-window-prob metric for comparison.
"""
import os, numpy as np, torch, torchvision, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
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

print("\n===== VAL streaming eval (n=%d, pos=%d) =====" % (len(y), int(y.sum())))
sweep(maxp,  "max-window-prob (matches training)")
sweep(maxema, "max-EMA (streaming/deployment rule)")
sweep_recall_first(maxema, "max-EMA (deployment)")
