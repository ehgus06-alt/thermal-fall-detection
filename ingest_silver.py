"""
Ingest the 'silver' dataset (Google-Drive split zips with Split/Class/clip/frame.jpg)
into the cache, tagged source='silver'. Handles clips whose frames are split across the
zips by indexing all three globally first. Same Lepton-emulation preprocessing as training.
Writes cache/{Fall,NonFall}/SILVER_*.npy and cache/manifest_silver.json.

NOTE: the source zips are capped at 65535 entries each (non-ZIP64), so a few frames past
that boundary are unreadable; clips are still ~complete and MIN_FRAMES filters short ones.
"""
import os, re, io, json, zipfile, collections
import numpy as np
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

SRC   = r"D:\써멀\DATASET_KHAN"
ZIPS  = [os.path.join(SRC, z) for z in (
    "drive-download-20260819T143305Z-1-001.zip",
    "drive-download-20260819T143305Z-1-002.zip",
    "drive-download-20260819T143305Z-1-003.zip")]
HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
TARGET, MIN_FRAMES = 128, 30

def natnum(s):
    m = re.search(r'(\d+)', os.path.basename(s)); return int(m.group(1)) if m else -1

def lepton_from_pil(im, target=TARGET):
    a = np.asarray(im.convert('L'), dtype=np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi - lo < 1e-3: hi = lo + 1.0
    a = np.clip((a - lo) / (hi - lo), 0, 1) * 255.0
    im = Image.fromarray(a.astype(np.uint8)); w, h = im.size; s = max(w, h)
    c = Image.new('L', (s, s), 0); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(c.resize((target, target), Image.BILINEAR), dtype=np.uint8)

def sanitize(x):
    return re.sub(r'[^0-9A-Za-z_]', '_', x)

def process_clip(args):
    cls, out_name, frames = args   # frames: list of (zip_path, member_name) already sorted
    zh = {}
    stack = np.empty((len(frames), TARGET, TARGET), dtype=np.uint8)
    try:
        for i, (zp, name) in enumerate(frames):
            if zp not in zh:
                zh[zp] = zipfile.ZipFile(zp)
            stack[i] = lepton_from_pil(Image.open(io.BytesIO(zh[zp].read(name))))
    except Exception as e:
        for z in zh.values(): z.close()
        return (out_name, 0, f'error:{type(e).__name__}')
    for z in zh.values(): z.close()
    outdir = os.path.join(CACHE, cls); os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, out_name + '.npy'), stack)
    return (out_name, len(stack), 'ok')

def main():
    os.makedirs(CACHE, exist_ok=True)
    # global index: (cls, split, clip) -> list of (framenum, zip, member)
    index = collections.defaultdict(list)
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as zf:
            for n in zf.namelist():
                if not n.lower().endswith('.jpg'):
                    continue
                parts = n.split('/')
                if len(parts) < 4:
                    continue
                split, cls, clip, frame = parts[0], parts[1], parts[2], parts[-1]
                if cls not in ('Fall', 'NonFall'):
                    continue
                index[(cls, split, clip)].append((natnum(frame), zp, n))
    print(f"indexed {len(index)} clips from {len(ZIPS)} zips", flush=True)

    tasks = []
    for (cls, split, clip), frames in index.items():
        if len(frames) < MIN_FRAMES:
            continue
        frames.sort(key=lambda x: x[0])
        out_name = f"SILVER_{sanitize(split)[:3]}_{sanitize(clip)}"
        tasks.append((cls, out_name, [(zp, n) for _, zp, n in frames]))
    print(f"ingesting {len(tasks)} clips (>= {MIN_FRAMES} frames)", flush=True)

    manifest, done = [], 0
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(process_clip, t): t for t in tasks}
        for fu in as_completed(futs):
            cls = futs[fu][0]
            name, n, status = fu.result()
            done += 1
            if status == 'ok':
                manifest.append(dict(cls=cls, clip=name, n=int(n),
                                     label=1 if cls == 'Fall' else 0, source='silver'))
            if done % 50 == 0 or status != 'ok':
                print(f"[{done}/{len(tasks)}] {name} n={n} {status}", flush=True)
    manifest.sort(key=lambda r: r['clip'])
    with open(os.path.join(CACHE, 'manifest_silver.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    nf = sum(r['label'] for r in manifest); nn = len(manifest) - nf
    print(f"done. SILVER cached: Fall={nf} NonFall={nn}, frames={sum(r['n'] for r in manifest)}", flush=True)

if __name__ == '__main__':
    main()
