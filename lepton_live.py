"""
Real-time fall detection for a FLIR Lepton (PureThermal / UVC).

    python lepton_live.py --camera 0            # live, colorized-RGB Lepton output
    python lepton_live.py --camera 0 --y16       # live, raw 16-bit radiometric output
    python lepton_live.py --simclip Fall/Fall29  # NO CAMERA: replay a cached clip through
                                                 # the identical live loop (logic test)

Preprocessing matches training exactly: grayscale -> per-frame 2-98% percentile stretch
(cancels Lepton AGC swings) -> letterbox square -> 128. A rolling W-frame buffer feeds the
same 3-ch (gray/MHI/MEI) window into MobileNetV3. EMA-smoothed prob triggers an alarm with
a cooldown; each alarm saves a snapshot.
"""
import os, time, argparse, collections, json, urllib.request, threading
import numpy as np
import cv2
import torch, torchvision, torch.nn as nn
from PIL import Image
from dataset import make_window_feat

HERE  = os.path.dirname(os.path.abspath(__file__))
CKPT  = os.path.join(HERE, "runs", "best.pt")
CACHE = os.path.join(HERE, "cache")

# ═══════════════════════════════════════════════════════════════════════════
#  백엔드 주소: 이 한 줄만 당신 서버 주소로 바꾸세요 (또는 실행 시 --webhook 로 지정)
WEBHOOK_URL = ""      # 예: "https://your-site.com/api/fall"
# ═══════════════════════════════════════════════════════════════════════════

# ---------- preprocessing (identical to cache_frames.py) ----------
def prep_gray128(gray_u8, target=128):
    a = gray_u8.astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    a = np.clip((a - lo) / max(hi - lo, 1e-3), 0, 1) * 255.0
    im = Image.fromarray(a.astype(np.uint8)); w, h = im.size; s = max(w, h)
    c = Image.new('L', (s, s), 0); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(c.resize((target, target), Image.BILINEAR), np.uint8)

def frame_to_gray(frame, y16):
    if y16:                       # raw 16-bit thermal -> 8-bit via percentile AGC
        f = frame.astype(np.float32)
        lo, hi = np.percentile(f, 1), np.percentile(f, 99)
        return np.clip((f - lo) / max(hi - lo, 1e-3), 0, 1) * 255
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame

# ---------- model ----------
def load_model(device):
    ck = torch.load(CKPT, map_location=device)
    m = torchvision.models.mobilenet_v3_small()
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)
    m.load_state_dict(ck['model']); m.to(device).eval()
    return m, ck['args']['W']

class Detector:
    def __init__(self, model, W, device, thr, ema_a):
        self.m, self.W, self.dev, self.thr, self.a = model, W, device, thr, ema_a
        self.buf = collections.deque(maxlen=W); self.ema = 0.0
    def push(self, gray128):
        self.buf.append(gray128)
        if len(self.buf) < self.W:
            return None, self.ema
        stack = np.stack(self.buf)                        # [W,128,128]
        feat = torch.from_numpy(make_window_feat(stack, 0, self.W))[None].to(self.dev)
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=(self.dev == 'cuda')):
            p = torch.sigmoid(self.m(feat).squeeze()).item()
        self.ema = self.a * p + (1 - self.a) * self.ema
        return p, self.ema

# ---------- frame sources ----------
def camera_backends():
    # Windows: PureThermal streams via MSMF (DSHOW fails). Linux (Raspberry Pi): V4L2.
    if os.name == 'nt':
        return [(cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_ANY, 'ANY')]
    return [(cv2.CAP_V4L2, 'V4L2'), (cv2.CAP_ANY, 'ANY')]

def camera_source(idx, y16):
    cap = None
    for be, name in camera_backends():
        c = cv2.VideoCapture(int(idx), be)
        if y16:
            c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        else:
            c.set(cv2.CAP_PROP_CONVERT_RGB, 1)     # RGB888 colorized
        time.sleep(0.8)
        ok = False; fr = None
        for _ in range(30):
            ok, fr = c.read()
            if ok and fr is not None:
                break
            time.sleep(0.05)
        if ok and fr is not None:
            print(f"camera opened via {name}, frame={fr.shape}")
            cap = c; break
        c.release()
    if cap is None:
        raise RuntimeError(f"cannot pull frames from camera {idx}. "
                           f"Close the FLIR Lepton app (one app at a time), then retry. "
                           f"Diagnose with: python list_cameras.py")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        yield frame
    cap.release()

def post_json(url, payload, timeout=4):
    """POST a JSON payload in a background thread so detection never blocks on the network."""
    def _send():
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(url, data=data,
                                         headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=timeout) as r:
                print(f"    webhook -> {url} [{r.status}]")
        except Exception as e:
            print(f"    webhook FAILED -> {url}: {e}")
    threading.Thread(target=_send, daemon=True).start()

def blob_centroid_y(gray, pct=88):
    """Vertical centroid (row) of the warm blob = brightest pixels. None if none found."""
    rows = np.where(gray >= np.percentile(gray, pct))[0]
    return float(rows.mean()) if rows.size else None

def recent_descent(cy_hist, height):
    """Largest downward drop of the blob centroid over recent history, as a fraction of frame height.
    Fast fall -> big drop within the short window; slow lie-down -> small drop."""
    vals = [c for c in cy_hist if c is not None]
    if len(vals) < 2:
        return 0.0
    mn, best = vals[0], 0.0
    for c in vals:
        mn = min(mn, c); best = max(best, c - mn)
    return best / height

def posture_aspect(gray, pct=92, min_pixels=25):
    """Width/height of the warm blob. >1 = wide (lying), <1 = tall (standing). None if no blob.
    Works best on a fixed live feed with a hot person over a cool room."""
    m = gray >= np.percentile(gray, pct)
    r, c = np.where(m)
    if r.size < min_pixels:
        return None
    return (c.max() - c.min() + 1) / (r.max() - r.min() + 1)

def simclip_source(clip):
    cls, name = clip.split('/')
    stack = np.load(os.path.join(CACHE, cls, name + '.npy'))   # already 128 gray
    for g in stack:
        yield g            # already preprocessed; bypass prep below

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=str, default=None)
    ap.add_argument('--simclip', type=str, default=None)
    ap.add_argument('--y16', action='store_true')
    ap.add_argument('--thr', type=float, default=0.47)
    ap.add_argument('--ema', type=float, default=0.4)
    ap.add_argument('--stride', type=int, default=1, help='run model every N frames')
    ap.add_argument('--persist', type=int, default=3, help='windows EMA must stay >=thr before alarm (kills brief blips)')
    ap.add_argument('--cooldown', type=float, default=5.0, help='seconds between alarms')
    ap.add_argument('--min-descent', type=float, default=0.0,
                    help='OFF by default (0). Optional per-camera gate: require this much fast blob drop '
                         '(frac of height) to allow an alarm. Does NOT generalize across viewpoints '
                         '(a value that helps one camera blocks real falls on another) - calibrate per fixed setup.')
    ap.add_argument('--display', action='store_true', help='show HUD window (FALL / LIED / SAFE)')
    ap.add_argument('--lie-aspect', type=float, default=1.4,
                    help='blob width/height above this = LIED (lying) state shown in orange. Tune to your camera using the w/h shown on the HUD.')
    ap.add_argument('--webhook', default=WEBHOOK_URL, help='backend URL for the sustained-fall JSON (default = WEBHOOK_URL at top of file)')
    ap.add_argument('--fall-hold', type=float, default=3.0, help='seconds a FALL must stay held before the backend signal is sent')
    ap.add_argument('--device-id', default='lepton-01', help='device id included in webhook payload')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, W = load_model(device)
    det = Detector(model, W, device, args.thr, args.ema)
    snap_dir = os.path.join(HERE, 'alarms'); os.makedirs(snap_dir, exist_ok=True)
    print(f"device={device} W={W} thr={args.thr} ema={args.ema}  (Ctrl-C to stop)")

    sim = args.simclip is not None
    src = simclip_source(args.simclip) if sim else camera_source(args.camera, args.y16)
    last_alarm = -1e9; n = 0; t0 = time.time(); fired_any = False; over = 0
    fall_latched = False; upright_count = 0    # FALL stays latched until person stands up again
    fall_start = 0.0; webhook_sent = False; fall_prob = 0.0   # sustained-fall backend signal
    cy_hist = collections.deque(maxlen=15)     # blob centroid history for descent gate
    aspect_hist = collections.deque(maxlen=8)  # blob w/h history for LIED posture state
    for i, frame in enumerate(src):
        gray128 = frame if sim else prep_gray128(frame_to_gray(frame, args.y16))
        if i % args.stride:
            continue
        p, ema = det.push(gray128)
        n += 1
        now = time.time()
        cy_hist.append(blob_centroid_y(gray128))
        descent = recent_descent(cy_hist, gray128.shape[0])
        over = over + 1 if (p is not None and ema >= args.thr) else 0
        # descent gate: model says fall AND (optionally) a fast downward drop
        alarm = (over >= args.persist and now - last_alarm > args.cooldown
                 and descent >= args.min_descent)

        # posture + FALL-latch state machine.
        # LIED vs FALL is NOT about the pose (both end wide/horizontal) but about HOW you got
        # there: a fall fires the model -> FALL latched until you stand up again; a slow
        # lie-down never fires -> LIED. Aspect only tells us "is the person still down".
        wh = posture_aspect(gray128)
        if wh is not None:
            aspect_hist.append(wh)
        med_wh = float(np.median(aspect_hist)) if aspect_hist else 0.0
        lying = med_wh >= args.lie_aspect
        if alarm:
            if not fall_latched:                  # entering FALL: start the hold timer
                fall_start = now; webhook_sent = False; fall_prob = float(ema)
            fall_latched = True; upright_count = 0
        elif not lying:
            upright_count += 1
            if upright_count >= 8:                # sustained upright -> person got back up
                fall_latched = False
        else:
            upright_count = 0
        state = 'FALL' if fall_latched else ('LIED' if lying else 'SAFE')

        # backend signal: only when the FALL has been HELD for --fall-hold seconds (fell & still down)
        if fall_latched and not webhook_sent and (now - fall_start) >= args.fall_hold:
            webhook_sent = True
            held = round(now - fall_start, 1)
            print(f"  >>> FALL held {held}s -> backend signal {'sent' if args.webhook else '(no URL set)'}")
            if args.webhook:
                post_json(args.webhook, dict(
                    event='fall', device_id=args.device_id,
                    timestamp=time.strftime('%Y-%m-%dT%H:%M:%S'), unix_time=round(now, 3),
                    held_seconds=held, probability=round(fall_prob, 3), frame=int(i)))

        if alarm:
            last_alarm = now; fired_any = True
            ts = time.strftime('%H:%M:%S')
            # NOTE: cv2.imwrite fails silently on non-ASCII (Korean) paths -> use PIL
            snap = os.path.join(snap_dir, f"fall_{time.strftime('%Y%m%d_%H%M%S')}_{i}.png")
            Image.fromarray(gray128).resize((320, 320), Image.NEAREST).save(snap)
            print(f"  [{ts}] >>> FALL detected <<<  frame={i} p={p:.2f} ema={ema:.2f}  (hold {args.fall_hold}s for backend signal)")
        elif p is not None and (ema > 0.3 or p > 0.5):
            print(f"  frame={i:4d} p={p:.2f} ema={ema:.2f} state={state}")

        if args.display:
            col = {'FALL': (0, 0, 255), 'LIED': (0, 165, 255), 'SAFE': (0, 200, 0)}[state]
            label = state
            vis = cv2.cvtColor(cv2.resize(gray128, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            bar = int((ema if p is not None else 0) * 384)
            cv2.rectangle(vis, (0, 378), (bar, 384), col, -1)
            cv2.putText(vis, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            cv2.putText(vis, f"p={ema:.2f}  w/h={med_wh:.2f} (LIED>={args.lie_aspect})",
                        (12, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow('lepton-fall', vis)
            if cv2.waitKey(20 if sim else 1) & 0xFF == ord('q'):
                break
    fps = n / max(time.time() - t0, 1e-6)
    print(f"processed {n} windows @ {fps:.1f} win/s. result: "
          f"{'FALL detected' if fired_any else 'no fall'}")
    if args.display: cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
