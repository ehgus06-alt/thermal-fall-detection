"""
Data audit for the thermal fall-detection dataset.
- Global stats: clip counts, frames/clip, resolution consistency (all clips).
- Motion analysis on a sample: locate where the person actually appears.
- Save montages so we can eyeball label quality.
Uses only PIL + numpy (no cv2/matplotlib).
"""
import os, re, json, sys
import numpy as np
from PIL import Image

ROOT = r"D:\써멀\THERMAL_ADL_FALL_LABEL\Thermal"
OUT  = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\audit_out"
os.makedirs(OUT, exist_ok=True)

DS = (160, 120)  # downscale for motion analysis (Lepton-like)

def natkey(s):
    m = re.search(r'(\d+)', s)
    return int(m.group(1)) if m else -1

def list_clips(cls):
    d = os.path.join(ROOT, cls)
    clips = sorted([c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))], key=natkey)
    return clips

def clip_frames(cls, clip):
    d = os.path.join(ROOT, cls, clip)
    fs = [f for f in os.listdir(d) if f.lower().endswith('.jpg')]
    fs.sort(key=natkey)
    return [os.path.join(d, f) for f in fs]

def load_gray_ds(path, size=DS):
    im = Image.open(path).convert('L').resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32)

# ---------- 1. Global stats over ALL clips ----------
print("=== GLOBAL STATS (all clips) ===", flush=True)
stats = {}
res_counter = {}
for cls in ['Fall', 'NonFall']:
    clips = list_clips(cls)
    counts = []
    for c in clips:
        frs = clip_frames(cls, c)
        counts.append(len(frs))
        if frs:  # record resolution of first frame
            with Image.open(frs[0]) as im:
                res_counter[im.size] = res_counter.get(im.size, 0) + 1
    counts = np.array(counts)
    stats[cls] = dict(n_clips=len(clips),
                      frames_total=int(counts.sum()),
                      frames_min=int(counts.min()), frames_max=int(counts.max()),
                      frames_mean=float(counts.mean()), frames_median=float(np.median(counts)))
    print(f"{cls}: {stats[cls]}", flush=True)
print("Resolutions (first-frame of each clip):", res_counter, flush=True)

# ---------- 2. Motion analysis + montage on a SAMPLE ----------
def analyze_clip(cls, clip, stride=3):
    frs = clip_frames(cls, clip)
    if len(frs) < 4:
        return None
    idxs = list(range(0, len(frs), stride))
    grays = [load_gray_ds(frs[i]) for i in idxs]
    # motion energy = mean abs diff vs previous sampled frame
    motion = [0.0]
    for k in range(1, len(grays)):
        motion.append(float(np.mean(np.abs(grays[k] - grays[k-1]))))
    motion = np.array(motion)
    peak_local = int(np.argmax(motion))
    peak_frame_idx = idxs[peak_local]
    return dict(n=len(frs), sampled=len(idxs), idxs=idxs, grays=grays,
                motion=motion, peak_frame=peak_frame_idx,
                motion_max=float(motion.max()), motion_mean=float(motion.mean()),
                frs=frs)

def make_montage(cls, clip, info, ncols=6):
    """Grid of evenly spaced frames + the peak-motion frame, at DS resolution (Lepton preview)."""
    frs = info['frs']; n = info['n']
    picks = sorted(set([int(x) for x in np.linspace(0, n-1, ncols)] + [info['peak_frame']]))
    tiles = []
    for i in picks:
        im = Image.open(frs[i]).convert('RGB').resize((160, 120), Image.BILINEAR)
        tiles.append((i, im))
    tw, th = 160, 120
    rows = 1
    cols = len(tiles)
    canvas = Image.new('RGB', (cols*tw, rows*th), (0,0,0))
    for j,(i,im) in enumerate(tiles):
        canvas.paste(im, (j*tw, 0))
    canvas.save(os.path.join(OUT, f"montage_{cls}_{clip}.png"))

SAMPLE = {'Fall': ['Fall0','Fall1','Fall50','Fall100','Fall150','Fall200','Fall257'],
          'NonFall': ['NonFall0','NonFall1','NonFall45','NonFall89']}

print("\n=== MOTION SAMPLE ===", flush=True)
motion_summary = []
for cls, clips in SAMPLE.items():
    for clip in clips:
        if not os.path.isdir(os.path.join(ROOT, cls, clip)):
            continue
        info = analyze_clip(cls, clip)
        if info is None:
            continue
        make_montage(cls, clip, info)
        row = dict(cls=cls, clip=clip, n=info['n'], peak_frame=info['peak_frame'],
                   motion_max=round(info['motion_max'],2), motion_mean=round(info['motion_mean'],2))
        motion_summary.append(row)
        print(row, flush=True)

with open(os.path.join(OUT, 'audit_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(dict(stats=stats, resolutions={str(k):v for k,v in res_counter.items()},
                   motion_summary=motion_summary), f, indent=2, ensure_ascii=False)
print(f"\nSaved montages + audit_summary.json to {OUT}", flush=True)
