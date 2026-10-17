
"""Clean-split retraining + threshold re-selection + evaluation (MediScan audit study)."""
import os, sys, json, time, argparse, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score

ROOT = r"D:\CapStone-2"
OUT  = os.path.join(ROOT, "clean_study")
RES  = os.path.join(OUT, "results"); os.makedirs(RES, exist_ok=True)
sys.path.insert(0, ROOT)
from train_tumornet import TumorNet, MultiClassFocalLoss, get_warmup_cosine_scheduler  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------- datasets
EVAL_TF = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                              transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
TRAIN_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(0.15, 0.15), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

class DS(Dataset):
    def __init__(self, paths, labels, tf):
        self.p, self.y, self.tf = paths, labels, tf
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        return self.tf(Image.open(self.p[i]).convert("RGB")), int(self.y[i])

class Focal(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ls=0.04):
        super().__init__(); self.alpha, self.gamma, self.ls = alpha, gamma, ls
    def forward(self, x, y):
        ce = F.cross_entropy(x, y, reduction="none", label_smoothing=self.ls, weight=self.alpha)
        return (((1 - torch.exp(-ce)) ** self.gamma) * ce).mean()

def focal_bin(logit, y, alpha, gamma=2.0, ls=0.04):
    ce = F.cross_entropy(logit, y, reduction="none", label_smoothing=ls, weight=alpha)
    return (((1 - torch.exp(-ce)) ** gamma) * ce).mean()

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha); idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def build(arch, ncls):
    if arch == "dual":
        m = TumorNet(num_classes=ncls); return m, "DualPathNet"
    if arch == "r18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, ncls); return m, "ResNet-18"
    if arch == "r50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, ncls); return m, "ResNet-50"
    raise ValueError(arch)

# ---------------------------------------------------------------- eval / metrics
def tta_probs(model, ds, bs=64):
    model.eval(); outs = []
    ld = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)
    def fwd(x):
        with torch.no_grad():
            return torch.softmax(model(x.to(DEV)), 1).cpu().numpy()
    for x, _ in ld:
        a = fwd(x)
        b = fwd(torch.flip(x, [3]))
        c = fwd(transforms.functional.center_crop(x, [200, 200]) if False else torch.nn.functional.interpolate(
            transforms.functional.center_crop(x, [200, 200]), size=(224, 224), mode="bilinear", align_corners=False))
        d = fwd(torch.flip(torch.nn.functional.interpolate(
            transforms.functional.center_crop(x, [200, 200]), size=(224, 224), mode="bilinear", align_corners=False), [3]))
        outs.append((a + b + c + d) / 4.0)
    return np.concatenate(outs, 0)

def boot_ci(y, p, metric, n=1000, seed=0):
    rng = np.random.RandomState(seed); vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        try: vals.append(metric(y[i], p[i]))
        except Exception: pass
    if not vals: return (float("nan"), float("nan"))
    return tuple(np.percentile(vals, [2.5, 97.5]))

def eval_multi(y, prob):
    pred = prob.argmax(1); C = prob.shape[1]
    acc = float((pred == y).mean()); f1 = float(f1_score(y, pred, average="macro", zero_division=0))
    try: auc = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
    except Exception: auc = float("nan")
    per = {}
    for c in range(C):
        tp = int(((pred == c) & (y == c)).sum()); fn = int(((pred != c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum()); tn = int(((pred != c) & (y != c)).sum())
        per[c] = dict(support=int((y == c).sum()), tp=tp, fn=fn, fp=fp, tn=tn,
                      recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1),
                      f1=2 * tp / max(2 * tp + fp + fn, 1))
    return dict(accuracy=acc, macro_f1=f1, auc=auc,
                acc_ci=boot_ci(y, pred, lambda a, b: float((a == b).mean())),
                per_class=per, n=int(len(y)))

def eval_bin(y, p, tau):
    pred = (p >= tau).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    try: auc = float(roc_auc_score(y, p))
    except Exception: auc = float("nan")
    return dict(threshold=float(tau), accuracy=float((pred == y).mean()), sensitivity=sens, specificity=spec,
                precision=prec, f1=f1, auc=auc, tp=tp, fn=fn, fp=fp, tn=tn, n=int(len(y)),
                acc_ci=boot_ci(y, pred, lambda a, b: float((a == b).mean())),
                sens_ci=boot_ci(y, p, lambda a, b: float(((b >= tau).astype(int) == a)[a == 1].mean())),
                spec_ci=boot_ci(y, p, lambda a, b: float(((b >= tau).astype(int) == a)[a == 0].mean())))

def pick_tau(y, p, min_rate=0.90):
    best = (0.5, -1)
    for t in np.arange(0.05, 0.951, 0.025):
        pred = (p >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
        tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
        sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
        if min(sens, spec) >= min_rate and t > best[0]:
            best = (float(t), min(sens, spec))
    if best[1] < 0:  # infeasible -> max balanced accuracy
        for t in np.arange(0.05, 0.951, 0.025):
            pred = (p >= t).astype(int)
            tp = ((pred == 1) & (y == 1)).sum(); fn = ((pred == 0) & (y == 1)).sum()
            tn = ((pred == 0) & (y == 0)).sum(); fp = ((pred == 1) & (y == 0)).sum()
            sc = tp / max(tp + fn, 1) + tn / max(tn + fp, 1)
            if sc > best[1]: best = (float(t), sc)
    return best[0]

def sweep(y, p):
    rows = []
    for t in [0.3, 0.4, 0.5, 0.58, 0.7, 0.78, 0.85]:
        r = eval_bin(y, p, t); rows.append({k: r[k] for k in ("threshold", "accuracy", "sensitivity", "specificity", "precision", "f1")})
    return rows

# ---------------------------------------------------------------- training
def train_model(arch, tr_p, tr_y, va_p, va_y, ncls, epochs, lr, tag, bs=32):
    model, name = build(arch, ncls)
    model = model.to(DEV)
    tr_ds = DS(tr_p, tr_y, TRAIN_TF); va_ds = DS(va_p, va_y, EVAL_TF)
    cnt = np.bincount(tr_y, minlength=ncls).astype(np.float32)
    cw = 1.0 / np.maximum(cnt, 1); sw = [cw[t] for t in tr_y]
    sampler = WeightedRandomSampler(sw, num_samples=len(tr_ds), replacement=True)
    tl = DataLoader(tr_ds, batch_size=bs, sampler=sampler, num_workers=6, pin_memory=True, drop_last=True)
    vl = DataLoader(va_ds, batch_size=64, shuffle=False, num_workers=6, pin_memory=True)
    if ncls == 4:
        alpha = torch.tensor([1.35, 1.05, 0.90, 0.90], device=DEV)
        crit = MultiClassFocalLoss(alpha=alpha, gamma=1.5, label_smoothing=0.05)
    else:
        crit = None
        alpha = torch.tensor([1.65, 1.00], device=DEV) if cnt[1] < cnt[0] else torch.tensor([1.00, 1.65], device=DEV)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch = get_warmup_cosine_scheduler(opt, 2, epochs, lr)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEV.type == "cuda"))
    best = (-1, None); hist = []; patience = 0
    for ep in range(1, epochs + 1):
        model.train(); tl_sum = n = 0; t0 = time.time()
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEV.type == "cuda")):
                if ncls == 4 and np.random.rand() < 0.5 and ep < int(epochs * 0.65):
                    xm, ya, yb, lam = mixup(x, y); out = model(xm)
                    loss = lam * crit(out, ya) + (1 - lam) * crit(out, yb)
                elif ncls == 2 and np.random.rand() < 0.5 and ep < int(epochs * 0.65):
                    xm, ya, yb, lam = mixup(x, y); out = model(xm)
                    loss = lam * focal_bin(out, ya, alpha) + (1 - lam) * focal_bin(out, yb, alpha)
                else:
                    out = model(x)
                    loss = crit(out, y) if ncls == 4 else focal_bin(out, y, alpha)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tl_sum += float(loss) * len(y); n += len(y)
        sch.step()
        # validation
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for x, y in vl:
                with torch.amp.autocast("cuda", enabled=(DEV.type == "cuda")):
                    o = model(x.to(DEV))
                ps.append(torch.softmax(o.float(), 1).cpu().numpy()); ys.append(y.numpy())
        P = np.concatenate(ps); Y = np.concatenate(ys)
        if ncls == 4:
            sc = float(f1_score(Y, P.argmax(1), average="macro", zero_division=0))
        else:
            sc = float(roc_auc_score(Y, P[:, 1])) if len(np.unique(Y)) > 1 else 0.0
        hist.append(dict(epoch=ep, loss=tl_sum / max(n, 1), val_score=sc, sec=round(time.time() - t0, 1)))
        print(f"  [{tag}] ep{ep:02d} loss {tl_sum/max(n,1):.4f} val {sc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        if sc > best[0] + 1e-4:
            best = (sc, copy.deepcopy(model.state_dict())); patience = 0
        else:
            patience += 1
            if patience >= 4:
                print(f"  [{tag}] early stop at epoch {ep}", flush=True); break
    model.load_state_dict(best[1])
    torch.save({"model_state": model.state_dict(), "arch": arch, "num_classes": ncls}, os.path.join(RES, f"{tag}.pth"))
    return model, name, hist, best[0]

def load_ckpt(arch, ncls, path):
    ck = torch.load(path, map_location="cpu")
    model, name = build(arch, ncls)
    model.load_state_dict(ck["model_state"]); return model.to(DEV), name

# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)      # mri|cxr|gastro|smnr|evalonly
    ap.add_argument("--arch", default="dual")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sp = json.load(open(os.path.join(OUT, "clean_splits.json")))
    torch.manual_seed(42); np.random.seed(42)
    res = {}
    if a.task == "evalonly":
        # frozen original checkpoint on original + clean test
        spec = {"mri": ("dual", 4, "TumorNet_best.pth"), "cxr": ("dual", 2, "PneumoNet_best.pth"),
                "gastro": ("dual", 2, "GastroNet_best.pth")}
        for m, (arch, ncls, ck) in spec.items():
            model, name = load_ckpt(arch, ncls, os.path.join(ROOT, "checkpoints", ck))
            r = {}
            for split in ("test_all", "test"):
                p = sp[m][split]; y = np.array(sp[m][split + "_labels"])
                P = tta_probs(model, DS(p, y, EVAL_TF))
                if ncls == 4:
                    r[split] = eval_multi(y, P)
                else:
                    pv = sp[m]["val"]; yv = np.array(sp[m]["val_labels"])
                    Pv = tta_probs(model, DS(pv, yv, EVAL_TF))
                    tau_clean = pick_tau(yv, Pv[:, 1])
                    r[split] = eval_bin(y, P[:, 1], tau_clean)
                    r[split]["tau_from_clean_val"] = tau_clean
                    r[split]["sweep"] = sweep(y, P[:, 1])
            res[m] = r
            print(m, json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("accuracy", "macro_f1", "auc", "sensitivity", "specificity", "f1", "tau_from_clean_val", "n")} for k, v in r.items()}, indent=1), flush=True)
        json.dump(res, open(os.path.join(RES, "frozen_on_clean.json"), "w"), indent=1)
        print("DONE_EVAL"); return

    if a.task == "smnr":
        g = sp["gastro"]; sm = sp["smnr"]
        tr_p = g["paths"] + sm["smnr_neg"]; tr_y = g["labels"] + [0] * len(sm["smnr_neg"])
        model, name, hist, bv = train_model(a.arch, tr_p, tr_y, g["val"], g["val_labels"], 2, a.epochs or 22, a.lr or 3e-4, "smnr", 32)
        pv = np.array(g["val_labels"]); Pv = tta_probs(model, DS(g["val"], pv, EVAL_TF))
        tau = pick_tau(pv, Pv[:, 1])
        pt = np.array(g["test_labels"]); Pt = tta_probs(model, DS(g["test"], pt, EVAL_TF))
        res["test_clean"] = eval_bin(pt, Pt[:, 1], tau)
        yc = np.array([0] * len(sm["control"]) + [1] * len(sm["eso"]))
        Pc = tta_probs(model, DS(sm["control"] + sm["eso"], yc, EVAL_TF))
        res["control_heldout"] = eval_bin(yc, Pc[:, 1], 0.5)
        res["history"] = hist; res["best_val"] = bv
        json.dump(res, open(os.path.join(RES, f"smnr_{a.arch}.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in res.items() if k != "history"}, indent=1)); print("DONE_SMNR"); return

    ep_map = {"mri": 16, "cxr": 20, "gastro": 22}; lr_map = {"mri": 4e-4, "cxr": 3e-4, "gastro": 3e-4}
    d = sp[a.task]; ncls = len(set(d["labels"]))
    tag = f"{a.task}_{a.arch}"
    model, name, hist, bv = train_model(a.arch, d["paths"], d["labels"], d["val"], d["val_labels"], ncls,
                                       a.epochs or ep_map[a.task], a.lr or lr_map[a.task], tag)
    r = {}
    for split in ("test_all", "test"):
        p = d[split]; y = np.array(d[split + "_labels"])
        P = tta_probs(model, DS(p, y, EVAL_TF))
        if ncls == 4:
            r[split] = eval_multi(y, P)
        else:
            yv = np.array(d["val_labels"]); Pv = tta_probs(model, DS(d["val"], yv, EVAL_TF))
            tau = pick_tau(yv, Pv[:, 1])
            r[split] = eval_bin(y, P[:, 1], tau); r[split]["tau_from_clean_val"] = tau
            r[split]["sweep"] = sweep(y, P[:, 1])
            r[split]["val"] = eval_bin(yv, Pv[:, 1], tau)
    r["history"] = hist; r["best_val"] = bv; r["arch"] = name
    json.dump(r, open(os.path.join(RES, f"{tag}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ("history",)}, indent=1)); print("DONE_TRAIN")

if __name__ == "__main__":
    main()
