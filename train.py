"""
MIL fall-detection training on Lepton-emulated windows.
Backbone: MobileNetV3-small (ImageNet pretrained) -> 1 logit per window.
Clip logit = max over its windows (noisy-OR MIL). Clip-level BCE with pos_weight.
Eval: clip-level Average Precision + best-F1 threshold (precision/recall/F1).
"""
import os, json, argparse, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
import torchvision

from dataset import ClipBagDataset, collate_bags, load_manifest

OUT = r"D:\써멀\THERMAL_ADL_FALL_LABEL\fall_ai\runs"

def build_model():
    m = torchvision.models.mobilenet_v3_small(weights=torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, 1)
    return m

def bag_logits(model, feats_list, device):
    """feats_list: list of [K_i,3,H,W]. Returns clip logits [B] via max-pool, and all-window logits."""
    sizes = [f.shape[0] for f in feats_list]
    x = torch.cat(feats_list, 0).to(device, non_blocking=True)
    wl = model(x).squeeze(1)                       # [sumK]
    clip = []
    off = 0
    for k in sizes:
        clip.append(wl[off:off+k].max())
        off += k
    return torch.stack(clip)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for feats, labels, _ in loader:
        cl = bag_logits(model, feats, device)
        ps.append(torch.sigmoid(cl).cpu().numpy())
        ys.append(labels.numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    ap = average_precision_score(y, p)
    # best-F1 threshold sweep
    best = (0, 0.5, 0, 0)
    for t in np.linspace(0.05, 0.95, 19):
        pr, rc, f1, _ = precision_recall_fscore_support(y, (p >= t).astype(int),
                                                        average='binary', zero_division=0)
        if f1 > best[0]:
            best = (f1, t, pr, rc)
    return dict(ap=float(ap), f1=float(best[0]), thr=float(best[1]),
                precision=float(best[2]), recall=float(best[3]), n=len(y))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--bs', type=int, default=8)
    ap.add_argument('--K', type=int, default=8)
    ap.add_argument('--W', type=int, default=24)
    ap.add_argument('--stride', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    man = load_manifest()
    labels = [r['label'] for r in man]
    tr, va = train_test_split(man, test_size=0.2, stratify=labels, random_state=args.seed)
    print(f"train {len(tr)} / val {len(va)} clips | "
          f"train pos={sum(r['label'] for r in tr)} val pos={sum(r['label'] for r in va)}", flush=True)

    tr_ds = ClipBagDataset(tr, W=args.W, stride=args.stride, K=args.K, train=True)
    va_ds = ClipBagDataset(va, W=args.W, stride=args.stride, train=False)
    tr_ld = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=0, collate_fn=collate_bags)
    va_ld = DataLoader(va_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_bags)

    model = build_model().to(device)
    npos = sum(r['label'] for r in tr); nneg = len(tr) - npos
    pos_weight = torch.tensor([nneg / max(npos, 1)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler('cuda')

    best_ap, log = -1, []
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time(); tot = 0.0
        for feats, labels, _ in tr_ld:
            labels = labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                cl = bag_logits(model, feats, device)
                loss = crit(cl, labels)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item() * len(labels)
        sched.step()
        m = evaluate(model, va_ld, device)
        m.update(epoch=ep, train_loss=tot / len(tr), sec=round(time.time() - t0, 1))
        log.append(m)
        print(f"ep{ep:02d} loss={m['train_loss']:.3f} AP={m['ap']:.3f} "
              f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} "
              f"thr={m['thr']:.2f} ({m['sec']}s)", flush=True)
        if m['ap'] > best_ap:
            best_ap = m['ap']
            torch.save(dict(model=model.state_dict(), args=vars(args), metrics=m),
                       os.path.join(OUT, 'best.pt'))
    with open(os.path.join(OUT, 'train_log.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(args=vars(args), log=log, best_ap=best_ap), f, indent=2)
    print(f"done. best val AP={best_ap:.3f} -> {os.path.join(OUT,'best.pt')}", flush=True)

if __name__ == '__main__':
    main()
