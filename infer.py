"""
Sliding-window fall inference with EMA smoothing + alarm threshold.

Modes:
  --clip Fall/Fall10   : offline test on a cached clip (no camera / no cv2 needed).
  --camera 0           : live FLIR Lepton via UVC (needs opencv-python + PureThermal).

Live path mirrors training preprocessing exactly (grayscale percentile-stretch ->
letterbox square -> 128), keeps a rolling buffer of W frames, and fires when the
EMA of window fall-probability stays above --thr.
"""
import os, argparse, json, collections
import numpy as np
import torch
import torchvision, torch.nn as nn

from dataset import make_window_feat, IMAGENET_MEAN, IMAGENET_STD
CACHE = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\cache"
CKPT  = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\runs\best.pt"

def load_model(device):
    ck = torch.load(CKPT, map_location=device)
    m = torchvision.models.mobilenet_v3_small()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    m.load_state_dict(ck['model']); m.to(device).eval()
    return m, ck

def stream_probs(model, stack, W, stride, device):
    """Yield (frame_index, window_prob) over a cached uint8 stack."""
    starts = range(0, max(1, len(stack) - W + 1), stride)
    for s in starts:
        feat = torch.from_numpy(make_window_feat(stack, s, W))[None].to(device)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(device == 'cuda')):
            p = torch.sigmoid(model(feat).squeeze()).item()
        yield s + W, p

def run_offline(args, device):
    model, ck = load_model(device)
    W = ck['args']['W']; thr = args.thr if args.thr else ck['metrics']['thr']
    cls, clip = args.clip.split('/')
    stack = np.load(os.path.join(CACHE, cls, clip + '.npy'))
    ema, alpha, fired = 0.0, args.ema, False
    print(f"clip={args.clip} frames={len(stack)} W={W} thr={thr:.2f}")
    for fidx, p in stream_probs(model, stack, W, args.stride, device):
        ema = alpha * p + (1 - alpha) * ema
        flag = 'FALL!' if ema >= thr else ''
        if ema >= thr and not fired:
            fired = True; flag = '>>> FALL ALARM <<<'
        if p > 0.3 or flag:
            print(f"  frame~{fidx:4d}  p={p:.2f}  ema={ema:.2f}  {flag}")
    print("RESULT:", "FALL detected" if fired else "no fall")

def run_camera(args, device):
    import cv2  # needs opencv-python
    from PIL import Image
    def lepton_prep(frame_bgr, target=128):
        g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lo, hi = np.percentile(g, 2), np.percentile(g, 98)
        g = np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1) * 255
        im = Image.fromarray(g.astype(np.uint8)); w, h = im.size; s = max(w, h)
        c = Image.new('L', (s, s), 0); c.paste(im, ((s - w) // 2, (s - h) // 2))
        return np.asarray(c.resize((target, target)), np.uint8)
    model, ck = load_model(device); W = ck['args']['W']
    thr = args.thr if args.thr else ck['metrics']['thr']
    cap = cv2.VideoCapture(int(args.camera))
    buf = collections.deque(maxlen=W); ema = 0.0
    print(f"live Lepton W={W} thr={thr:.2f} (q to quit)")
    while True:
        ok, frame = cap.read()
        if not ok: break
        buf.append(lepton_prep(frame))
        if len(buf) == W:
            stack = np.stack(buf)
            feat = torch.from_numpy(make_window_feat(stack, 0, W))[None].to(device)
            with torch.no_grad():
                p = torch.sigmoid(model(feat).squeeze()).item()
            ema = args.ema * p + (1 - args.ema) * ema
            if ema >= thr:
                cv2.putText(frame, "FALL", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.putText(frame, f"ema={ema:.2f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('lepton-fall', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', type=str, help='offline: e.g. Fall/Fall10')
    ap.add_argument('--camera', type=str, help='live: UVC index e.g. 0')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--ema', type=float, default=0.4)
    ap.add_argument('--thr', type=float, default=0.0, help='0 = use trained best-F1 thr')
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.clip: run_offline(args, device)
    elif args.camera: run_camera(args, device)
    else: print("specify --clip Fall/Fall10  or  --camera 0")
