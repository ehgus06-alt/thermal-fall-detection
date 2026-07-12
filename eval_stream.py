"""
Deployment-realistic evaluation: run the streaming EMA detector over each VAL clip
(same split as train.py) and score clip-level detection. Sweeps EMA threshold and
also reports the raw max-window-prob metric for comparison.
"""
import os, numpy as np, torch, torchvision, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from dataset import make_window_feat, load_manifest
CACHE = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\cache"
CKPT  = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\runs\best.pt"

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

print("\n===== VAL streaming eval (n=%d, pos=%d) =====" % (len(y), int(y.sum())))
sweep(maxp,  "max-window-prob (matches training)")
sweep(maxema, "max-EMA (streaming/deployment rule)")
