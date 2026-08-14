"""
MIL (multiple-instance) dataset over cached Lepton-emulated clip stacks.

A clip = a "bag" of short windows. Each window -> 3-channel image:
  ch0 = last-frame gray   (posture / appearance)
  ch1 = MHI               (motion history: recency+trajectory -> fall sweeps downward)
  ch2 = MEI               (motion energy: where movement happened)
Window inherits the clip label; MIL max-pool at train time finds the fall sub-segment,
so pre-fall "walking" windows in a Fall clip don't have to be individually labeled.
"""
import os, json, math
import numpy as np
import torch
from torch.utils.data import Dataset

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)

def load_manifest():
    """Merge all dataset manifests, tagging each clip with its source."""
    man = []
    with open(os.path.join(CACHE, 'manifest.json'), encoding='utf-8') as f:
        for r in json.load(f):
            r.setdefault('source', 'flir')
            man.append(r)
    for extra, src in [('manifest_khan.json', 'khan'), ('manifest_mine.json', 'mine')]:
        path = os.path.join(CACHE, extra)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                for r in json.load(f):
                    r.setdefault('source', src)
                    man.append(r)
    return man

def window_starts(n, W, stride):
    if n <= W:
        return [0]
    return list(range(0, n - W + 1, stride))

def augment_stack(seg):
    """Train-time augmentation on a [W,H,Wd] float32 window (0-255), applied CONSISTENTLY
    across all frames so motion (MHI/MEI) stays coherent. No vertical flip (gravity matters)."""
    if np.random.random() < 0.5:                     # h-flip (falls go left or right)
        seg = seg[:, :, ::-1]
    if np.random.random() < 0.7:                     # translation (camera position)
        dy, dx = np.random.randint(-12, 13), np.random.randint(-12, 13)
        seg = np.roll(seg, (dy, dx), axis=(1, 2))
        if dy > 0: seg[:, :dy, :] = 0
        elif dy < 0: seg[:, dy:, :] = 0
        if dx > 0: seg[:, :, :dx] = 0
        elif dx < 0: seg[:, :, dx:] = 0
    if np.random.random() < 0.7:                     # brightness / contrast (thermal auto-gain)
        seg = np.clip(seg * np.random.uniform(0.8, 1.2) + np.random.uniform(-20, 20), 0, 255)
    if np.random.random() < 0.2:                     # polarity invert (white-hot <-> black-hot)
        seg = 255.0 - seg
    if np.random.random() < 0.3:                     # soft blur (Lepton low-res emulation)
        seg = (seg + np.roll(seg, 1, 1) + np.roll(seg, -1, 1)
               + np.roll(seg, 1, 2) + np.roll(seg, -1, 2)) / 5.0
    if np.random.random() < 0.5:                     # sensor noise
        seg = np.clip(seg + np.random.normal(0, np.random.uniform(2, 10), seg.shape), 0, 255)
    return seg.astype(np.float32)

def make_window_feat(stack, s, W, motion_thr=12, augment=False):
    """stack: uint8 [N,H,W]; returns float32 [3,H,W] normalized with ImageNet stats."""
    seg = stack[s:s+W].astype(np.float32)            # [W,H,Wd]
    H, Wd = seg.shape[1], seg.shape[2]
    if seg.shape[0] < W:                              # pad short (edge clips) by repeat
        seg = np.concatenate([seg, np.repeat(seg[-1:], W - seg.shape[0], 0)], 0)
    if augment:
        seg = augment_stack(seg)
    diffs = np.abs(np.diff(seg, axis=0))             # [W-1,H,Wd]
    mask = diffs > motion_thr
    # MHI: decay from W..0
    mhi = np.zeros((H, Wd), np.float32)
    for k in range(mask.shape[0]):
        mhi = np.where(mask[k], float(W), np.maximum(mhi - 1.0, 0.0))
    mhi /= float(W)
    mei = diffs.max(axis=0) / 255.0                  # motion energy
    gray = seg[-1] / 255.0                           # last frame
    feat = np.stack([gray, mhi, mei], axis=-1)       # [H,Wd,3]
    feat = (feat - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(feat, (2, 0, 1)).copy()      # [3,H,W]

class ClipBagDataset(Dataset):
    """Returns a bag of K windows per clip. Train: random K. Eval: evenly-spaced up to max_eval."""
    def __init__(self, records, W=24, stride=8, K=8, train=True, max_eval=16, augment=False):
        self.records = records
        self.W, self.stride, self.K = W, stride, K
        self.train, self.max_eval = train, max_eval
        self.augment = augment
        self._cache = {}

    def _stack(self, rec):
        key = (rec['cls'], rec['clip'])
        if key not in self._cache:
            self._cache[key] = np.load(os.path.join(CACHE, rec['cls'], rec['clip'] + '.npy'))
        return self._cache[key]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        stack = self._stack(rec)
        starts = window_starts(len(stack), self.W, self.stride)
        if self.train:
            if len(starts) >= self.K:
                idx = np.random.choice(len(starts), self.K, replace=False)
            else:
                idx = np.random.choice(len(starts), self.K, replace=True)
            chosen = [starts[j] for j in idx]
        else:
            if len(starts) > self.max_eval:
                sel = np.linspace(0, len(starts) - 1, self.max_eval).round().astype(int)
                chosen = [starts[j] for j in sel]
            else:
                chosen = starts
        aug = self.augment and self.train
        feats = np.stack([make_window_feat(stack, s, self.W, augment=aug) for s in chosen], 0)  # [K,3,H,W]
        return torch.from_numpy(feats), torch.tensor(float(rec['label'])), i

def collate_bags(batch):
    """Variable K (eval) -> keep as list of tensors + labels."""
    feats = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch])
    idxs = torch.tensor([b[2] for b in batch])
    return feats, labels, idxs
