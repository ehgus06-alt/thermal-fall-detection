"""
Ingest the KHAN/TSF dataset (zipped clips) into our cache, tagged source='khan'.
- Fall*.zip  -> Fall clip  (label 1)
- ADL*.zip   -> NonFall clip (label 0)   [ADL = activities of daily living]
Skips corrupt zips, junk (non-Fall/ADL), and duplicate copies of the same clip id.
Frames are read straight from the zip and run through the SAME Lepton-emulation
preprocessing as cache_frames.py (grayscale + percentile stretch + letterbox 128).
Writes cache/{Fall,NonFall}/KHAN_*.npy and cache/manifest_khan.json.
"""
import os, re, io, json, zipfile
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC   = r"D:\써멀\DATASET_KHAN"
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TARGET, MIN_FRAMES = 128, 30

def natkey(s):
    m = re.search(r'(\d+)', os.path.basename(s)); return int(m.group(1)) if m else -1

def lepton_from_pil(im, target=TARGET):
    a = np.asarray(im.convert('L'), dtype=np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi - lo < 1e-3: hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
    im = Image.fromarray(a.astype(np.uint8))
    w, h = im.size; s = max(w, h)
    c = Image.new('L', (s, s), 0); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(c.resize((target, target), Image.BILINEAR), dtype=np.uint8)

def process_zip(args):
    zpath, cls, clip = args
    try:
        with zipfile.ZipFile(zpath) as zf:
            if zf.testzip() is not None:
                return (clip, 0, 'corrupt')
            names = sorted([n for n in zf.namelist()
                            if n.lower().endswith(('.jpg', '.jpeg', '.png'))], key=natkey)
            if len(names) < MIN_FRAMES:
                return (clip, len(names), 'short')
            stack = np.empty((len(names), TARGET, TARGET), dtype=np.uint8)
            for i, n in enumerate(names):
                stack[i] = lepton_from_pil(Image.open(io.BytesIO(zf.read(n))))
    except Exception as e:
        return (clip, 0, f'error:{type(e).__name__}')
    outdir = os.path.join(CACHE, cls); os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, clip + '.npy'), stack)
    return (clip, len(stack), 'ok')

def main():
    os.makedirs(CACHE, exist_ok=True)
    tasks, seen = [], set()
    for z in sorted(os.listdir(SRC)):
        if not z.lower().endswith('.zip'):
            continue
        m = re.match(r'(fall|adl)\s*\(?(\d+)', z.lower())
        if not m:
            print(f"skip junk: {z}"); continue
        kind, num = m.group(1), int(m.group(2))
        key = (kind, num)
        if key in seen:
            print(f"skip duplicate: {z}"); continue
        zpath = os.path.join(SRC, z)
        if not zipfile.is_zipfile(zpath):
            print(f"skip corrupt: {z}"); continue
        seen.add(key)
        cls = 'Fall' if kind == 'fall' else 'NonFall'
        clip = f"KHAN_{'Fall' if kind=='fall' else 'ADL'}{num}"
        tasks.append((zpath, cls, clip))

    print(f"\ningesting {len(tasks)} KHAN clips -> {CACHE}", flush=True)
    manifest, done = [], 0
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_zip, t): t for t in tasks}
        for fu in as_completed(futs):
            _, cls, clip = futs[fu]
            name, n, status = fu.result()
            done += 1
            if status == 'ok':
                manifest.append(dict(cls=cls, clip=clip, n=int(n),
                                     label=1 if cls == 'Fall' else 0, source='khan'))
            print(f"[{done}/{len(tasks)}] {clip} n={n} {status}", flush=True)
    manifest.sort(key=lambda r: (r['cls'], natkey(r['clip'])))
    with open(os.path.join(CACHE, 'manifest_khan.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    nf = sum(r['label'] for r in manifest); nn = len(manifest) - nf
    print(f"\ndone. KHAN cached: Fall={nf} NonFall={nn}, frames={sum(r['n'] for r in manifest)}", flush=True)

if __name__ == '__main__':
    main()
