"""
Stage 1 preprocessing: decode every clip's frames -> Lepton-emulated grayscale
-> cache as uint8 .npy per clip (N x H x W). Fast to reload for windowed training.

Lepton emulation:
  - convert FLIR iron-palette RGB -> grayscale luminance (Lepton is single-channel thermal)
  - per-frame percentile stretch (2-98%)  -> cancels FLIR auto-gain brightness jumps
  - letterbox to square (preserve up/down orientation; portrait & landscape unified)
  - downscale to TARGET (loses detail -> approximates Lepton's low resolution)
"""
import os, re, json
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT   = r"D:\써멀\THERMAL_ADL_FALL_LABEL\Thermal"
CACHE  = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\cache"
TARGET = 128          # cached square size
MIN_FRAMES = 30       # drop clips shorter than this (e.g. NonFall65=12)

def natkey(s):
    m = re.search(r'(\d+)', s); return int(m.group(1)) if m else -1

def lepton_frame(path, target=TARGET):
    im = Image.open(path).convert('L')          # -> single-channel thermal proxy
    a = np.asarray(im, dtype=np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)   # per-frame contrast stretch
    if hi - lo < 1e-3: hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
    im = Image.fromarray(a.astype(np.uint8))
    # letterbox to square (pad with 0 = cold), keep orientation
    w, h = im.size
    s = max(w, h)
    canvas = Image.new('L', (s, s), 0)
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((target, target), Image.BILINEAR)  # downscale -> Lepton-ish
    return np.asarray(canvas, dtype=np.uint8)

def clip_frame_paths(cls, clip):
    d = os.path.join(ROOT, cls, clip)
    fs = [f for f in os.listdir(d) if f.lower().endswith('.jpg')]
    fs.sort(key=natkey)
    return [os.path.join(d, f) for f in fs]

def process_clip(args):
    cls, clip = args
    paths = clip_frame_paths(cls, clip)
    if len(paths) < MIN_FRAMES:
        return (cls, clip, 0, 'skipped_short')
    stack = np.empty((len(paths), TARGET, TARGET), dtype=np.uint8)
    for i, p in enumerate(paths):
        stack[i] = lepton_frame(p)
    outdir = os.path.join(CACHE, cls)
    os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, clip + '.npy'), stack)
    return (cls, clip, len(paths), 'ok')

def main():
    os.makedirs(CACHE, exist_ok=True)
    tasks = []
    for cls in ['Fall', 'NonFall']:
        d = os.path.join(ROOT, cls)
        clips = sorted([c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))], key=natkey)
        tasks += [(cls, c) for c in clips]
    print(f"caching {len(tasks)} clips -> {CACHE}", flush=True)
    manifest = []
    done = 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(process_clip, t) for t in tasks]
        for fu in as_completed(futs):
            cls, clip, n, status = fu.result()
            done += 1
            if status == 'ok':
                manifest.append(dict(cls=cls, clip=clip, n=int(n),
                                     label=1 if cls == 'Fall' else 0))
            if done % 25 == 0 or status != 'ok':
                print(f"[{done}/{len(tasks)}] {cls}/{clip} n={n} {status}", flush=True)
    manifest.sort(key=lambda r: (r['cls'], natkey(r['clip'])))
    with open(os.path.join(CACHE, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    nf = sum(1 for r in manifest if r['label'] == 1)
    nn = sum(1 for r in manifest if r['label'] == 0)
    print(f"done. cached clips: Fall={nf} NonFall={nn}, total frames={sum(r['n'] for r in manifest)}", flush=True)

if __name__ == '__main__':
    main()
