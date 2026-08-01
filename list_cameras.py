"""Enumerate cameras so you can find the FLIR Lepton (PureThermal) index.
Lepton 3.5 = 160x120 (often reported 160x122 with 2 telemetry rows).
Cross-platform: Windows (DSHOW/MSMF) and Linux / Raspberry Pi (V4L2, /dev/video*).
Run after plugging in the camera:  python list_cameras.py
"""
import os, glob, cv2

if os.name == 'nt':
    backend, bname = cv2.CAP_DSHOW, 'DSHOW'
else:
    backend, bname = cv2.CAP_V4L2, 'V4L2'
    vids = sorted(glob.glob('/dev/video*'))
    print(f"/dev/video* devices: {vids or 'none'}")

print(f"scanning camera indices 0..9 with {bname} ...")
found = []
for i in range(10):
    cap = cv2.VideoCapture(i, backend)
    if cap.isOpened():
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        shape = frame.shape if ok else None
        hint = "  <-- likely LEPTON" if (w, h) in ((160, 120), (160, 122)) or \
               (shape and shape[:2] in ((120, 160), (122, 160))) else ""
        print(f"[index {i}] opened  {w}x{h}  read_ok={ok} frame={shape}{hint}")
        found.append(i)
        cap.release()
print(f"\nopened indices: {found}")
print("Use the Lepton index with:  python lepton_live.py --camera <idx>")
