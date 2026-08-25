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
from datetime import datetime
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

def open_camera(idx, y16):
    """Try each backend and return an opened VideoCapture (verified with one test frame), or None."""
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
            return c
        c.release()
    return None

def camera_source(idx, y16, reconnect_wait=3.0, tick=0.2):
    """Never-dying camera generator. Yields a frame when the Lepton is healthy, or None when it's
    down - so the caller keeps running and heartbeats thermal=FAIL to the backend. A missing or dead
    camera no longer crashes or silently stops the monitor. Reconnection runs in a BACKGROUND thread
    so opening a camera (slow on Windows: seconds per backend) never blocks the heartbeat. Ctrl-C to stop."""
    cap = {'c': None}; opening = {'busy': False}; announced = {'down': False}; last_try = [0.0]

    def try_open():
        c = open_camera(idx, y16)                   # slow; runs off the main loop
        cap['c'] = c; opening['busy'] = False
        if c is not None:
            print(f"camera {idx} UP"); announced['down'] = False

    while True:
        c = cap['c']
        if c is None:
            if not opening['busy'] and time.time() - last_try[0] >= reconnect_wait:
                last_try[0] = time.time(); opening['busy'] = True
                threading.Thread(target=try_open, daemon=True).start()
            if not announced['down']:
                print(f"camera {idx} DOWN -> thermal=FAIL heartbeats; retrying every {reconnect_wait:.0f}s "
                      f"(close the FLIR Lepton app if it is holding the device)")
                announced['down'] = True
            yield None                              # camera down: caller sends a FAIL heartbeat
            time.sleep(tick)
            continue
        ok, frame = c.read()
        if not ok or frame is None:                 # mid-run loss -> drop it and reconnect (in bg)
            print(f"camera {idx} read failed -> reconnecting")
            try: c.release()
            except Exception: pass
            cap['c'] = None; last_try[0] = time.time()
            yield None
            time.sleep(tick)
            continue
        yield frame
    cap.release()

# ═══════════════════════════════════════════════════════════════════════════
#  백엔드로 보낼 JSON (백엔드 확정 스키마).
#  state -> event_type 매핑: SAFE=안전, LIED(누움)=WARNING, FALL(낙상)=DANGER.
STATE_EVENT = {'SAFE': 'SAFE', 'LIED': 'WARNING', 'FALL': 'DANGER'}
# HUD 라벨: 상태 + 백엔드 심각도 병기
HUD_LABEL = {'SAFE': 'SAFE', 'LIED': 'LIED (WARNING)', 'FALL': 'FALL (DANGER)'}

def build_payload(*, device_id, seq, now, report_type, event_type, sensor_health,
                  battery_pct, rssi, uptime_sec):
    return {
        "device_id":   device_id,                        # 대상자 매칭용 (1~64자)
        "seq":         int(seq),                          # 전송 일련번호 (중복·순서 판별)
        "measured_at": datetime.fromtimestamp(now).astimezone().isoformat(timespec='seconds'),  # ISO8601 +TZ
        "report_type": report_type,                       # HEARTBEAT | EVENT
        "event_type":  event_type,                        # SAFE | WARNING | DANGER (EVENT 시 필수)
        "sensor_health": {                                # 센서 3개 생존 여부
            "vibrator": sensor_health['vibrator'],        # OK | FAIL | UNKNOWN
            "radar":    sensor_health['radar'],           # OK | FAIL | UNKNOWN
            "thermal":  sensor_health['thermal'],         # OK | FAIL | UNKNOWN
        },
        "device": {                                       # 원격 진동센서 노드 상태 (무선연결 예정, 현재 목업)
            "battery_pct": int(battery_pct),              # 0~100
            "rssi":        int(rssi),                     # <=0 (신호 세기, dBm)
            "uptime_sec":  int(uptime_sec),               # >=0 (재부팅 후 경과)
        },
    }
# ═══════════════════════════════════════════════════════════════════════════

_send_stats = {'ok': 0, 'fail': 0}
_send_lock = threading.Lock()

def post_json(url, payload, timeout=4, retries=0, backoff=1.0):
    """POST JSON in a background thread so detection never blocks on the network. Retries up to
    `retries` extra times with exponential backoff (1,2,4,... capped 8s) so a transient blip does
    not lose a critical EVENT (e.g. DANGER). seq is constant across retries -> backend can dedup.
    First-try successes are silent (heartbeats are frequent); retries and final failures are logged."""
    def _send():
        data = json.dumps(payload).encode('utf-8')
        seq = payload.get('seq')
        for attempt in range(retries + 1):
            try:
                req = urllib.request.Request(url, data=data,
                                             headers={'Content-Type': 'application/json'}, method='POST')
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    with _send_lock:
                        _send_stats['ok'] += 1
                    if attempt:
                        print(f"    webhook OK on retry {attempt} (seq={seq}) [{r.status}]")
                    return
            except Exception as e:
                if attempt < retries:
                    time.sleep(min(backoff * (2 ** attempt), 8.0))
                else:
                    with _send_lock:
                        _send_stats['fail'] += 1
                        fails = _send_stats['fail']
                    tries = 'tries' if retries else 'try'
                    print(f"    webhook FAILED (seq={seq}) after {retries + 1} {tries}: {e}  [total fails={fails}]")
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
    """Bounding box (x0,y0,x1,y1) of the person = the LARGEST warm-blob connected component among the
    hottest `pct`% pixels, or None if nothing big enough. Taking the biggest component (not the min/max
    of ALL hot pixels) keeps the box on the person and ignores stray hot pixels or distractors (a
    radiator, sunlit wall, electronics) that would otherwise inflate it well beyond the body."""
    m = (gray >= np.percentile(gray, pct)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))   # drop speckle noise
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))                  # largest blob (skip bg label 0)
    if stats[k, cv2.CC_STAT_AREA] < min_pixels:
        return None
    x, y = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
    w, h = int(stats[k, cv2.CC_STAT_WIDTH]), int(stats[k, cv2.CC_STAT_HEIGHT])
    return x, y, x + w - 1, y + h - 1

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
    ap.add_argument('--reconnect-wait', type=float, default=3.0,
                    help='seconds between camera reconnect attempts while the Lepton is down (thermal=FAIL)')
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
    ap.add_argument('--send-retries', type=int, default=3,
                    help='extra retry attempts (exponential backoff) for EVENT sends so a transient network '
                         'blip does not lose a DANGER alert; heartbeats are not retried (the next one covers it)')
    ap.add_argument('--send-backoff', type=float, default=1.0,
                    help='base seconds for EVENT send retry backoff (1,2,4,... capped at 8s)')
    ap.add_argument('--heartbeat', type=float, default=5.0,
                    help='seconds between keep-alive heartbeats to the backend (current status; proves the link is alive)')
    ap.add_argument('--state-hold', type=float, default=2.0,
                    help='a state must persist this many seconds before its event is sent (debounces spikes from sudden movement)')
    ap.add_argument('--person-min-pixels', type=int, default=60,
                    help='min warm-blob size (px) counted as a person (used only when --multi-suppress is on)')
    ap.add_argument('--multi-suppress', action='store_true',
                    help='OFF by default. When on, suppress DANGER while 2+ warm blobs are seen (geometry '
                         'unreliable with multiple people). Leave OFF for a single occupant: thermal blobs '
                         'split a lone person (bare skin vs cool clothes) into several, which would mask a '
                         'real fall. Only enable for genuinely multi-occupant rooms.')
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
    # sensor_health + device: MOCK until the remote mmWave vibration/radar node is wired in
    ap.add_argument('--health-vibrator', choices=['OK', 'FAIL', 'UNKNOWN'], default='UNKNOWN',
                    help='MOCK vibrator sensor health (remote mmWave node not attached yet -> UNKNOWN)')
    ap.add_argument('--health-radar', choices=['OK', 'FAIL', 'UNKNOWN'], default='UNKNOWN',
                    help='MOCK radar sensor health (remote mmWave node not attached yet -> UNKNOWN)')
    ap.add_argument('--health-thermal', choices=['OK', 'FAIL', 'UNKNOWN'], default='OK',
                    help='thermal (Lepton) sensor health; OK while frames are streaming')
    ap.add_argument('--battery-pct', type=int, default=100,
                    help='MOCK battery percent of the remote vibration-sensor node (0~100)')
    ap.add_argument('--rssi', type=int, default=-55,
                    help='MOCK wireless signal strength to the remote node, dBm (<=0)')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, W = load_model(device)
    det = Detector(model, W, device, args.thr, args.ema)
    snap_dir = os.path.join(HERE, 'alarms'); os.makedirs(snap_dir, exist_ok=True)
    print(f"device={device} W={W} thr={args.thr} ema={args.ema}  (Ctrl-C to stop)")

    sim = args.simclip is not None
    src = simclip_source(args.simclip) if sim else camera_source(args.camera, args.y16, args.reconnect_wait)
    last_alarm = -1e9; n = 0; t0 = time.time(); fired_any = False; over = 0
    fall_latched = False; upright_count = 0    # FALL stays latched until person stands up again
    cy_hist = collections.deque(maxlen=15)     # blob centroid history for descent gate
    aspect_hist = collections.deque(maxlen=8)  # blob w/h history for LIED posture state
    bbox_hist = collections.deque(maxlen=30)   # (t, aspect, top_y) history for bbox collapse-speed
    people_hist = collections.deque(maxlen=15) # warm-blob count history (~1s) to debounce multi-person
    # backend sender state: 5s heartbeat + immediate send whenever the debounced event changes
    pending_event = 'SAFE'; pending_since = t0; confirmed_event = 'SAFE'
    last_reported = None; last_send = 0.0; seq = 0; last_thermal = None
    state = 'SAFE'; multi_person = False        # frozen values reused while the camera is down
    health = {'vibrator': args.health_vibrator, 'radar': args.health_radar, 'thermal': args.health_thermal}
    print(f"backend={args.webhook or '(none: pass --webhook)'}  "
          f"heartbeat={args.heartbeat}s  state-hold={args.state_hold}s")
    print(f"sensor_health: vibrator={health['vibrator']}(mock) radar={health['radar']}(mock) "
          f"thermal=live[OK when frames arrive, FAIL when camera down]"
          f"  | device MOCK: battery={args.battery_pct}% rssi={args.rssi}dBm")
    for i, frame in enumerate(src):
        now = time.time()
        thermal_ok = frame is not None           # None = camera down (resilient source keeps us alive)
        if thermal_ok and (sim or i % args.stride == 0):
            gray128 = frame if sim else prep_gray128(frame_to_gray(frame, args.y16))
            p, ema = det.push(gray128)
            n += 1
            cy_hist.append(blob_centroid_y(gray128))
            descent = recent_descent(cy_hist, gray128.shape[0])
            # bbox collapse-speed: how fast the warm-blob box flattens (fall) vs eases down (lie-down)
            bb = person_bbox(gray128)
            if bb is not None:
                bbox_hist.append((now, bbox_aspect(bb), bb[1]))
            da, dtop = collapse_metrics(bbox_hist, gray128.shape[0], args.collapse_lookback)
            fast_collapse = da >= args.collapse_aspect or dtop >= args.collapse_drop
            people_hist.append(count_person_blobs(gray128, min_pixels=args.person_min_pixels))
            # only meaningful when --multi-suppress is on; off by default (a lone person splits into
            # several warm blobs, which would wrongly suppress a real fall)
            multi_person = args.multi_suppress and int(np.median(people_hist)) >= 2
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
                label = HUD_LABEL[state]                             # e.g. "FALL (DANGER)"
                vis = cv2.cvtColor(cv2.resize(gray128, (384, 384), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)
                if bb is not None:                                   # draw the bbox we measure collapse on
                    s = 384 / gray128.shape[0]
                    cv2.rectangle(vis, (int(bb[0] * s), int(bb[1] * s)), (int(bb[2] * s), int(bb[3] * s)),
                                  (0, 0, 255) if fast_collapse else (255, 200, 0), 1)
                bar = int((ema if p is not None else 0) * 384)
                cv2.rectangle(vis, (0, 378), (bar, 384), col, -1)
                # auto-fit the label: shrink the font so even "FALL (DANGER)" stays inside the 384px window
                scale = 1.4
                (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
                if tw > 384 - 24:
                    scale *= (384 - 24) / tw
                cv2.putText(vis, label, (12, 46), cv2.FONT_HERSHEY_SIMPLEX, scale, col, max(2, round(scale * 2)))
                cv2.putText(vis, f"people={int(np.median(people_hist))}{'  MULTI: FALL alerts OFF' if multi_person else ''}",
                            (12, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255) if multi_person else (255, 255, 255), 1)
                cv2.putText(vis, f"da={da:.2f} dtop={dtop:.2f}{'  COLLAPSE' if fast_collapse else ''}",
                            (12, 356), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(vis, f"p={ema:.2f}  w/h={med_wh:.2f} (LIED>={args.lie_aspect})",
                            (12, 372), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow('lepton-fall', vis)
                if cv2.waitKey(20 if sim else 1) & 0xFF == ord('q'):
                    break

        # ---- backend sender: runs EVERY tick so a camera outage still heartbeats thermal=FAIL ----
        # 5s heartbeat + immediate send when the debounced person-state changes (EVENT) or when the
        # thermal health flips (immediate HEARTBEAT). During an outage `state` is frozen at its last
        # known value and multi-person is unknown, so DANGER is not newly manufactured or suppressed.
        health['thermal'] = args.health_thermal if thermal_ok else 'FAIL'
        if not thermal_ok:
            multi_person = False
            if args.display:                        # camera-down splash so the window is not stale
                down = np.zeros((384, 384, 3), np.uint8)
                cv2.putText(down, "THERMAL FAIL", (34, 196), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
                cv2.putText(down, "camera down - reconnecting", (40, 236), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow('lepton-fall', down)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        inst_event = STATE_EVENT[state]
        if inst_event != pending_event:
            pending_event = inst_event; pending_since = now
        if pending_event != confirmed_event and (now - pending_since) >= args.state_hold:
            confirmed_event = pending_event
        report_event = 'SAFE' if (multi_person and confirmed_event == 'DANGER') else confirmed_event
        event_changed = report_event != last_reported
        health_changed = health['thermal'] != last_thermal
        if event_changed or health_changed or (now - last_send) >= args.heartbeat:
            report_type = 'EVENT' if event_changed else 'HEARTBEAT'   # sensor-health change stays a heartbeat
            if event_changed:
                tag = ' [multi-person: DANGER suppressed]' if (multi_person and confirmed_event == 'DANGER') else ''
                print(f"  >>> EVENT -> {report_event}{tag}  (backend {'sent' if args.webhook else 'skipped: no URL'})")
            elif health_changed:
                print(f"  >>> thermal={health['thermal']}  (heartbeat, backend {'sent' if args.webhook else 'skipped: no URL'})")
            if args.webhook:
                # EVENT (person-state change, incl. DANGER) retries hard; HEARTBEAT relies on the next tick
                retries = args.send_retries if report_type == 'EVENT' else 0
                post_json(args.webhook, build_payload(
                    device_id=args.device_id, seq=seq, now=now,
                    report_type=report_type, event_type=report_event, sensor_health=health,
                    battery_pct=args.battery_pct, rssi=args.rssi, uptime_sec=now - t0),
                    retries=retries, backoff=args.send_backoff)
                seq += 1
            last_reported = report_event; last_send = now; last_thermal = health['thermal']
    fps = n / max(time.time() - t0, 1e-6)
    print(f"processed {n} windows @ {fps:.1f} win/s. result: "
          f"{'FALL detected' if fired_any else 'no fall'}")
    if args.display: cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
