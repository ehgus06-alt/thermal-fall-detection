"""Headless capture test: confirm OpenCV receives Lepton RGB888 frames and runs our
pipeline. Saves a few real frames so we can SEE the feed. No GUI (head-less safe)."""
import os, time, argparse
import numpy as np, cv2
from PIL import Image
import torch
from lepton_live import prep_gray128, frame_to_gray, load_model, Detector

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "capture_test_out")
os.makedirs(OUT, exist_ok=True)

def open_cam(idx):
    for be, name in [(cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_DSHOW, 'DSHOW'), (cv2.CAP_ANY, 'ANY')]:
        cap = cv2.VideoCapture(idx, be)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        time.sleep(0.8)
        ok = False; fr = None
        for _ in range(30):
            ok, fr = cap.read()
            if ok and fr is not None:
                break
            time.sleep(0.05)
        if ok and fr is not None:
            print(f"[{name}] streaming OK  frame={fr.shape} dtype={fr.dtype}")
            return cap, name
        cap.release()
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--camera', type=int, default=0)
    ap.add_argument('--frames', type=int, default=120)
    args = ap.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, W = load_model(device)
    det = Detector(model, W, device, thr=0.47, ema_a=0.4)
    cap, be = open_cam(args.camera)
    if cap is None:
        print("!! no frames. Is the FLIR Lepton app still open? Close it and retry.")
        return
    saved = 0; probs = []; t0 = time.time()
    for i in range(args.frames):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"frame {i}: read failed"); continue
        gray = prep_gray128(frame_to_gray(frame, y16=False))
        p, ema = det.push(gray)
        if p is not None:
            probs.append(p)
        if i in (5, args.frames // 2, args.frames - 3) and saved < 3:
            Image.fromarray(gray).resize((256, 256), Image.NEAREST).save(os.path.join(OUT, f"live_{i}.png"))
            saved += 1
    dt = time.time() - t0; cap.release()
    print(f"captured {args.frames} frames in {dt:.1f}s ({args.frames/max(dt,1e-6):.1f} fps), backend={be}")
    if probs:
        pr = np.array(probs)
        print(f"fall-prob: min={pr.min():.2f} mean={pr.mean():.2f} max={pr.max():.2f}")
    print(f"saved {saved} preview frames -> {OUT}")

if __name__ == '__main__':
    main()
