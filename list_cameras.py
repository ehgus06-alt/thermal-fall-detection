"""Enumerate UVC cameras so you can find the FLIR Lepton (PureThermal) index.
Lepton 3.5 = 160x120. Run this after plugging in the camera:  python list_cameras.py
"""
import cv2
print("scanning camera indices 0..7 ...")
found = []
for i in range(8):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)   # DSHOW backend is most reliable on Windows
    if cap.isOpened():
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        shape = frame.shape if ok else None
        hint = "  <-- likely LEPTON (160x120)" if (w, h) == (160, 120) or (shape and shape[:2] == (120, 160)) else ""
        print(f"[index {i}] opened  {w}x{h}  read_ok={ok} frame={shape}{hint}")
        found.append(i)
        cap.release()
    else:
        print(f"[index {i}] not available")
print(f"\nopened indices: {found}")
print("Use the Lepton index with:  python lepton_live.py --camera <idx>")
