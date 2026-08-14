"""
Record your OWN thermal clips from the FLIR Lepton and add them straight into the
training cache (source='mine'), so you can fine-tune on your real environment.

    # record a fall clip (act it out, then press q or wait for --frames)
    python record_lepton.py --camera 0 --label fall   --name livingroom_fall1
    # record a normal-activity clip
    python record_lepton.py --camera 0 --label nonfall --name livingroom_adl1

Then:  python train.py --full --tag _final     # retrain including your clips
Uses the SAME preprocessing as training (grayscale + percentile stretch + letterbox 128).
Standalone (no torch): only numpy / cv2 / PIL.
"""
import os, time, argparse, json, re
import numpy as np, cv2
from PIL import Image

HERE  = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

def prep_gray128(frame, y16, target=128):
    if y16:
        g = frame.astype(np.float32)
    else:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) if frame.ndim == 3 else frame.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    g = np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1) * 255.0
    im = Image.fromarray(g.astype(np.uint8)); w, h = im.size; s = max(w, h)
    c = Image.new('L', (s, s), 0); c.paste(im, ((s - w) // 2, (s - h) // 2))
    return np.asarray(c.resize((target, target), Image.BILINEAR), np.uint8)

def open_cam(idx, y16):
    backends = ([(cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_ANY, 'ANY')]
                if os.name == 'nt' else [(cv2.CAP_V4L2, 'V4L2'), (cv2.CAP_ANY, 'ANY')])
    for be, name in backends:
        c = cv2.VideoCapture(int(idx), be)
        if y16:
            c.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        else:
            c.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        time.sleep(0.8)
        for _ in range(30):
            ok, fr = c.read()
            if ok and fr is not None:
                print(f"camera opened via {name}, frame={fr.shape}")
                return c
            time.sleep(0.05)
        c.release()
    raise RuntimeError(f"cannot open camera {idx}. Close the FLIR app; try: python list_cameras.py")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--label', choices=['fall', 'nonfall'], required=True)
    ap.add_argument('--name', required=True, help='clip name, e.g. livingroom_fall1')
    ap.add_argument('--frames', type=int, default=0, help='0 = record until you press q')
    ap.add_argument('--y16', action='store_true')
    args = ap.parse_args()

    cls = 'Fall' if args.label == 'fall' else 'NonFall'
    clip = 'MINE_' + re.sub(r'[^0-9A-Za-z_]', '_', args.name)
    cap = open_cam(args.camera, args.y16)
    frames, t0 = [], time.time()
    print(f"RECORDING [{args.label}] '{clip}'  —  press q to stop (or {args.frames} frames)")
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(prep_gray128(frame, args.y16))
        vis = cv2.resize(frames[-1], (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.putText(vis, f"REC {len(frames)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
        cv2.imshow('record (q=stop)', vis)
        if (cv2.waitKey(1) & 0xFF == ord('q')) or (args.frames and len(frames) >= args.frames):
            break
    cap.release(); cv2.destroyAllWindows()

    if len(frames) < 30:
        print(f"too short ({len(frames)} frames, need >=30). not saved."); return
    os.makedirs(os.path.join(CACHE, cls), exist_ok=True)
    np.save(os.path.join(CACHE, cls, clip + '.npy'), np.stack(frames))

    # upsert into manifest_mine.json
    mpath = os.path.join(CACHE, 'manifest_mine.json')
    man = json.load(open(mpath, encoding='utf-8')) if os.path.exists(mpath) else []
    man = [r for r in man if r['clip'] != clip]
    man.append(dict(cls=cls, clip=clip, n=len(frames),
                    label=1 if cls == 'Fall' else 0, source='mine'))
    json.dump(man, open(mpath, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"saved {len(frames)} frames -> cache/{cls}/{clip}.npy  ({len(frames)/max(time.time()-t0,1e-6):.1f} fps)")
    print(f"manifest_mine.json now has {len(man)} clip(s). Retrain:  python train.py --full --tag _final")

if __name__ == '__main__':
    main()
