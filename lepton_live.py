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

# ═══════════════════════════════════════════════════════════════════════════
#  백엔드로 보낼 JSON. state -> event_type 매핑: SAFE/LIED = 정상, FALL = 낙상.
#  (WARNING 은 현재 미사용 — LIED 를 WARNING 으로 올리려면 아래 매핑만 바꾸면 됨)
STATE_EVENT = {'SAFE': 'NORMAL', 'LIED': 'NORMAL', 'FALL': 'FALL_DETECTED'}

def build_payload(*, device_id, now, event_type, confidence, height_drop, heavy_vibration, posture):
    return {
        "device_id":        device_id,
        "timestamp":        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now)),  # ISO8601 UTC
        "event_type":       event_type,                    # NORMAL | WARNING | FALL_DETECTED
        "confidence_score": round(float(confidence), 3),   # AI 확신도 0.0 ~ 1.0
        "sensor_summary": {                                # 부가정보 (대시보드 표시용)
            "height_drop":     bool(height_drop),          # 급격한 고도 하락 (descent/dtop 기반)
            "heavy_vibration": bool(heavy_vibration),      # mmWave 진동센서 (현재 목업; 하드웨어 부착 시 실측)
            "posture":         posture,                    # "horizontal" | "vertical"
        },
    }
# ═══════════════════════════════════════════════════════════════════════════

# ── mmWave 진동센서 (현재 목업) ──────────────────────────────────────────────
# 실제 mmWave 모듈이 아직 없어서(부품 대기) 목업 bool 을 돌려줍니다. 하드웨어가 오면
# read() 안의 목업 분기를 실제 드라이버 읽기로 바꾸기만 하면 됩니다 (True/False 계약 유지).
class VibrationSensor:
    """'지금 바닥 진동이 있는가?' 를 yes/no 로 답한다. mmWave 부착 전까지는 목업.
    mode: off=항상 False | on=항상 True | random=가끔 튐 | on-fall=낙상 충격 때만 True."""
    def __init__(self, mode='on-fall', pulse_sec=2.0, seed=None):
        self.mode = mode; self.pulse_sec = pulse_sec
        self.rng = np.random.default_rng(seed); self._pulse_until = 0.0
    def pulse(self, now):
        """낙상 충격 같은 외부 이벤트가 목업 진동을 pulse_sec 초 동안 True 로 만든다."""
        self._pulse_until = max(self._pulse_until, now + self.pulse_sec)
    def read(self, now):
        if self.mode == 'on':      return True
        if self.mode == 'random':  return bool(self.rng.random() < 0.05)   # 가끔 한 번 튀는 노이즈
        if self.mode == 'on-fall': return now < self._pulse_until          # 낙상 직후에만 감지
        return False                                                       # 'off'
# ═══════════════════════════════════════════════════════════════════════════

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

def person_bbox(gray, pct=92, min_pixels=25):
    """Tight bounding box (x0,y0,x1,y1) around the warm blob (brightest pixels), or None if too small.
    Same thresholding as posture_aspect so FALL/LIED geometry stays consistent."""
    m = gray >= np.percentile(gray, pct)
    r, c = np.where(m)
    if r.size < min_pixels:
        return None
    return int(c.min()), int(r.min()), int(c.max()), int(r.max())

def bbox_aspect(bb):
    """width/height of a bbox. >1 = wide (lying), <1 = tall (standing)."""
    x0, y0, x1, y1 = bb
    return (x1 - x0 + 1) / (y1 - y0 + 1)

def collapse_metrics(bbox_hist, height, lookback):
    """How FAST the person's box is flattening over the last `lookback` seconds:
       da   = aspect (w/h) increase across the window   -> box going tall -> wide
       dtop = how far the TOP edge dropped, frac of height -> head/shoulders coming down
    A fall flattens fast (big da/dtop in <~0.5s); a controlled lie-down barely registers.
    bbox_hist holds (t, aspect, top_y) tuples; returns (0,0) until >=2 samples are in-window."""
    if len(bbox_hist) < 2:
        return 0.0, 0.0
    t_now = bbox_hist[-1][0]
    win = [x for x in bbox_hist if t_now - x[0] <= lookback]
    if len(win) < 2:
        win = list(bbox_hist)[-2:]
    da   = win[-1][1] - win[0][1]
    dtop = (win[-1][2] - win[0][2]) / max(height, 1)
    return max(da, 0.0), max(dtop, 0.0)

def count_person_blobs(gray, pct=92, min_pixels=60):
    """How many distinct warm blobs (people) are >= min_pixels. With 2+ people the single-blob
    geometry (bbox aspect / collapse) is meaningless, so the caller suppresses FALL alerts and
    only tracks. Thermal blobs are noisy (limbs split, pets/radiators show up), so the caller
    debounces this count over ~1s before acting on it."""
    m = (gray >= np.percentile(gray, pct)).astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return int(sum(1 for k in range(1, num) if stats[k, cv2.CC_STAT_AREA] >= min_pixels))

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
    ap.add_argument('--webhook', default=WEBHOOK_URL, help='backend URL for status JSON (heartbeat + events). default = WEBHOOK_URL at top of file')
    ap.add_argument('--device-id', default='pi_node_01', help='device id included in every backend payload')
    ap.add_argument('--heartbeat', type=float, default=5.0,
                    help='seconds between keep-alive heartbeats to the backend (current status; proves the link is alive)')
    ap.add_argument('--state-hold', type=float, default=2.0,
                    help='a state must persist this many seconds before its event is sent (debounces spikes from sudden movement)')
    ap.add_argument('--person-min-pixels', type=int, default=60,
                    help='min warm-blob size (px) counted as a person; 2+ people -> geometry unreliable -> FALL alerts suppressed (tracking only)')
    ap.add_argument('--collapse-lookback', type=float, default=0.6,
                    help='seconds of bbox history used to measure how fast the person box flattens')
    ap.add_argument('--collapse-aspect', type=float, default=0.8,
                    help='bbox width/height must widen by at least this within the lookback to count as a FAST (fall-like) collapse')
    ap.add_argument('--collapse-drop', type=float, default=0.18,
                    help='OR the bbox TOP edge drops by at least this fraction of frame height within the lookback')
    ap.add_argument('--collapse-trigger', action='store_true',
                    help='let a FAST bbox collapse latch FALL on its own (catches CNN misses). OFF by default: '
                         'collapse thresholds are viewpoint-dependent - calibrate per fixed camera by watching da/dtop on the '
                         'HUD, and they can false-fire on fast sit-downs. Leave off to keep the CNN as the sole trigger.')
    ap.add_argument('--vibration-mock', choices=['off', 'on', 'random', 'on-fall'], default='on-fall',
                    help='MOCK mmWave vibration sensor (hardware pending). off=always false, on=always true, '
                         'random=occasional blip, on-fall=fires for ~2s after a detected fall (realistic). '
                         'Swap VibrationSensor.read() for the real driver when the module arrives.')
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
    cy_hist = collections.deque(maxlen=15)     # blob centroid history for descent gate
    aspect_hist = collections.deque(maxlen=8)  # blob w/h history for LIED posture state
    bbox_hist = collections.deque(maxlen=30)   # (t, aspect, top_y) history for bbox collapse-speed
    people_hist = collections.deque(maxlen=15) # warm-blob count history (~1s) to debounce multi-person
    # backend sender state: 5s heartbeat + immediate send whenever the debounced event changes
    pending_event = 'NORMAL'; pending_since = t0; confirmed_event = 'NORMAL'
    last_reported = None; last_send = 0.0
    vib = VibrationSensor(args.vibration_mock)   # MOCK until the mmWave module is attached
    print(f"backend={args.webhook or '(none: pass --webhook)'}  "
          f"heartbeat={args.heartbeat}s  state-hold={args.state_hold}s")
    print(f"vibration sensor = MOCK ({args.vibration_mock})  <- swap VibrationSensor.read() for real mmWave")
    for i, frame in enumerate(src):
        gray128 = frame if sim else prep_gray128(frame_to_gray(frame, args.y16))
        if i % args.stride:
            continue
        p, ema = det.push(gray128)
        n += 1
        now = time.time()
        cy_hist.append(blob_centroid_y(gray128))
        descent = recent_descent(cy_hist, gray128.shape[0])
        # bbox collapse-speed: how fast the warm-blob box flattens (fall) vs eases down (lie-down)
        bb = person_bbox(gray128)
        if bb is not None:
            bbox_hist.append((now, bbox_aspect(bb), bb[1]))
        da, dtop = collapse_metrics(bbox_hist, gray128.shape[0], args.collapse_lookback)
        fast_collapse = da >= args.collapse_aspect or dtop >= args.collapse_drop
        people_hist.append(count_person_blobs(gray128, min_pixels=args.person_min_pixels))
        multi_person = int(np.median(people_hist)) >= 2   # debounced: 2+ people -> suppress FALL alerts
        over = over + 1 if (p is not None and ema >= args.thr) else 0
        # two independent FALL triggers, both respecting the cooldown:
        #   model_alarm    = CNN motion window says fall (+ optional descent gate)
        #   collapse_alarm = bbox flattened FAST (fall-like), only when --collapse-trigger is on
        model_alarm = over >= args.persist and descent >= args.min_descent
        collapse_alarm = args.collapse_trigger and fast_collapse
        alarm = (model_alarm or collapse_alarm) and now - last_alarm > args.cooldown

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
            fall_latched = True; upright_count = 0
        elif not lying:
            upright_count += 1
            if upright_count >= 8:                # sustained upright -> person got back up
                fall_latched = False
        else:
            upright_count = 0
        state = 'FALL' if fall_latched else ('LIED' if lying else 'SAFE')
        if state == 'FALL':
            vib.pulse(now)                       # mock: a fall's impact drives the vibration sensor
        vibration = vib.read(now)

        # ---- backend sender: 5s heartbeat + immediate send on a debounced state change ----
        # A state must persist --state-hold seconds to be 'confirmed', so spikes from sudden
        # movement never reach the backend. With 2+ people the geometry is unreliable, so FALL is
        # masked to NORMAL: the heartbeat keeps proving the link is alive but no false alert goes out.
        inst_event = STATE_EVENT[state]
        if inst_event != pending_event:
            pending_event = inst_event; pending_since = now
        if pending_event != confirmed_event and (now - pending_since) >= args.state_hold:
            confirmed_event = pending_event
        report_event = 'NORMAL' if (multi_person and confirmed_event == 'FALL_DETECTED') else confirmed_event
        changed = report_event != last_reported
        if changed or (now - last_send) >= args.heartbeat:
            if changed:
                tag = ' [multi-person: FALL suppressed]' if (multi_person and confirmed_event == 'FALL_DETECTED') else ''
                vibtag = ' vibration=ON' if vibration else ''
                print(f"  >>> event -> {report_event}{tag}{vibtag}  (backend {'sent' if args.webhook else 'skipped: no URL'})")
            if args.webhook:
                post_json(args.webhook, build_payload(
                    device_id=args.device_id, now=now, event_type=report_event, confidence=ema,
                    height_drop=max(descent, dtop) >= args.collapse_drop,
                    heavy_vibration=vibration,
                    posture='horizontal' if lying else 'vertical'))
            last_reported = report_event; last_send = now

        if alarm:
            last_alarm = now; fired_any = True
            ts = time.strftime('%H:%M:%S')
            trig = 'model' if model_alarm else 'collapse'
            pstr = f"{p:.2f}" if p is not None else "--"
            # NOTE: cv2.imwrite fails silently on non-ASCII (Korean) paths -> use PIL
            snap = os.path.join(snap_dir, f"fall_{time.strftime('%Y%m%d_%H%M%S')}_{i}.png")
            Image.fromarray(gray128).resize((320, 320), Image.NEAREST).save(snap)
            print(f"  [{ts}] >>> FALL detected <<<  frame={i} p={pstr} ema={ema:.2f} via={trig} "
                  f"da={da:.2f} dtop={dtop:.2f}  (needs {args.state_hold}s hold to alert backend)")
        elif p is not None and (ema > 0.3 or p > 0.5):
            print(f"  frame={i:4d} p={p:.2f} ema={ema:.2f} state={state} da={da:.2f} dtop={dtop:.2f}")

        if args.display:
            col = {'FALL': (0, 0, 255), 'LIED': (0, 165, 255), 'SAFE': (0, 200, 0)}[state]
            label = state
            vis = cv2.cvtColor(cv2.resize(gray128, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
            if bb is not None:                                   # draw the bbox we measure collapse on
                s = 384 / gray128.shape[0]
                cv2.rectangle(vis, (int(bb[0] * s), int(bb[1] * s)), (int(bb[2] * s), int(bb[3] * s)),
                              (0, 0, 255) if fast_collapse else (255, 200, 0), 1)
            bar = int((ema if p is not None else 0) * 384)
            cv2.rectangle(vis, (0, 378), (bar, 384), col, -1)
            cv2.putText(vis, label, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.4, col, 3)
            cv2.putText(vis, f"people={int(np.median(people_hist))}{'  MULTI: FALL alerts OFF' if multi_person else ''}",
                        (12, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255) if multi_person else (255, 255, 255), 1)
            cv2.putText(vis, f"da={da:.2f} dtop={dtop:.2f}{'  COLLAPSE' if fast_collapse else ''}",
                        (12, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
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
