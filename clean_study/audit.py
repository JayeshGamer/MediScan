
"""Cross-split leakage audit + clean-split construction (MediScan)."""
import os, sys, glob, json, hashlib
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = r"D:\CapStone-2"
OUT  = os.path.join(ROOT, "clean_study")
os.makedirs(OUT, exist_ok=True)
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def sha256_file(p, buf=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def px_hash(p):
    with Image.open(p) as im:
        return hashlib.sha1(im.convert("RGB").tobytes()).hexdigest()

def dhash(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    v = 0
    for b in bits: v = (v << 1) | int(b)
    return v

_D = np.cos(np.pi * (2 * np.arange(32)[:, None] + 1) * np.arange(32)[None, :] / 64.0) * np.sqrt(2 / 32)
_D[0, :] *= np.sqrt(0.5)
def phash(p):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), dtype=np.float64)
    c = _D @ a @ _D.T
    low = c[:8, :8].flatten()
    med = np.median(low[1:])
    bits = (low > med).astype(np.int64)
    v = 0
    for b in bits: v = (v << 1) | int(b)
    return v

def hamming(A, B, chunk=256):
    """A:(n,) uint64, B:(m,) uint64 -> (n,m) uint8 hamming distances."""
    A = np.ascontiguousarray(A, dtype=np.uint64); B = np.ascontiguousarray(B, dtype=np.uint64)
    out = np.empty((len(A), len(B)), dtype=np.uint8)
    for i in range(0, len(A), chunk):
        X = np.bitwise_xor(A[i:i+chunk, None], B[None, :])
        out[i:i+chunk] = POP[np.ascontiguousarray(X, dtype=np.uint64).view(np.uint8).reshape(len(X), -1, 8)].sum(2)
    return out

# ---------------------------------------------------------------- sample sets
def mri_samples():
    root = os.path.join(ROOT, "Brain Tumor MRI Dataset")
    classes = sorted(os.listdir(os.path.join(root, "Training")))
    tr_all = []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(os.path.join(root, "Training", c, "*"))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")): tr_all.append((f, ci))
    te = []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(os.path.join(root, "Testing", c, "*"))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")): te.append((f, ci))
    lbl = [s[1] for s in tr_all]
    tr, va = train_test_split(tr_all, test_size=0.15, stratify=lbl, random_state=42)
    return classes, tr, va, te

def cxr_samples():
    root = os.path.join(ROOT, "chest_xray")
    classes = sorted(os.listdir(os.path.join(root, "train")))
    tr_all = []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(os.path.join(root, "train", c, "*"))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")): tr_all.append((f, ci))
    te = []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(os.path.join(root, "test", c, "*"))):
            if f.lower().endswith((".jpg", ".jpeg", ".png")): te.append((f, ci))
    lbl = [s[1] for s in tr_all]
    tr, va = train_test_split(tr_all, test_size=0.15, stratify=lbl, random_state=42)
    return classes, tr, va, te

GASTRO_ROOT = os.path.join(ROOT, "labeled-images")
ULCER = ["upper-gi-tract/pathological-findings/esophagitis-a",
         "upper-gi-tract/pathological-findings/esophagitis-b-d",
         "upper-gi-tract/pathological-findings/barretts",
         "upper-gi-tract/pathological-findings/barretts-short-segment",
         "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1",
         "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2",
         "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-3",
         "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-1-2",
         "lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2-3"]
NORMAL = ["upper-gi-tract/anatomical-landmarks/pylorus",
          "upper-gi-tract/anatomical-landmarks/retroflex-stomach",
          "lower-gi-tract/anatomical-landmarks/cecum",
          "lower-gi-tract/anatomical-landmarks/retroflex-rectum"]

def gastro_samples():
    all_s = []
    for folder in NORMAL:
        for f in sorted(glob.glob(os.path.join(GASTRO_ROOT, folder, "*.jpg"))): all_s.append((f, 0))
    for folder in ULCER:
        for f in sorted(glob.glob(os.path.join(GASTRO_ROOT, folder, "*.jpg"))): all_s.append((f, 1))
    lbl = [s[1] for s in all_s]
    tv, te = train_test_split(all_s, test_size=0.15, stratify=lbl, random_state=42)
    tvl = [s[1] for s in tv]
    tr, va = train_test_split(tv, test_size=0.1765, stratify=tvl, random_state=42)
    return ["Normal", "Ulcer"], tr, va, te

def zline_samples():
    z = sorted(glob.glob(os.path.join(GASTRO_ROOT, "upper-gi-tract/anatomical-landmarks/z-line", "*.jpg")))
    eso = []
    for f in ["upper-gi-tract/pathological-findings/esophagitis-a",
              "upper-gi-tract/pathological-findings/esophagitis-b-d",
              "upper-gi-tract/pathological-findings/barretts",
              "upper-gi-tract/pathological-findings/barretts-short-segment"]:
        eso += sorted(glob.glob(os.path.join(GASTRO_ROOT, f, "*.jpg")))
    return z, eso

def hash_all(paths, tag):
    fn = os.path.join(OUT, f"hash_{tag}.npz")
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True)
        if len(d["paths"]) == len(paths) and list(d["paths"]) == list(paths):
            print(f"  [cache] {tag}: {len(paths)}")
            return d
    sha, px, dh, ph = [], [], [], []
    for i, p in enumerate(paths):
        sha.append(sha256_file(p)); px.append(px_hash(p)); dh.append(dhash(p)); ph.append(phash(p))
        if (i + 1) % 500 == 0: print(f"  {tag} {i+1}/{len(paths)}", flush=True)
    d = dict(paths=np.array(paths), sha=np.array(sha), px=np.array(px),
             dh=np.array(dh, dtype=np.uint64), ph=np.array(ph, dtype=np.uint64))
    np.savez(fn, **d)
    return d

def min_dist(H, ref_dh, ref_ph, drop_ref_self=False):
    if len(ref_dh) == 0: return np.full(len(H["dh"]), 255, dtype=np.uint16)
    d = np.minimum(hamming(H["dh"], ref_dh), hamming(H["ph"], ref_ph))
    d = d.astype(np.uint16)
    if drop_ref_self:
        idx = {p: j for j, p in enumerate(ref_paths)}
        for i, p in enumerate(H["paths"]):
            j = idx.get(p)
            if j is not None: d[i, j] = 255
    return d.min(1)

def dedup_within(dh, ph, thr=4):
    """Greedy grouping: keep first occurrence, drop later near-dups."""
    keep, groups = [], {}
    rep_dh, rep_ph, rep_idx = [], [], []
    order = np.arange(len(dh))
    for i in order:
        if rep_idx:
            dd = np.minimum(hamming(dh[i:i+1], np.array(rep_dh))[0], hamming(ph[i:i+1], np.array(rep_ph))[0])
            j = int(np.argmin(dd))
            if dd[j] <= thr:
                groups.setdefault(rep_idx[j], []).append(i); continue
        rep_dh.append(int(dh[i])); rep_ph.append(int(ph[i])); rep_idx.append(int(i)); keep.append(int(i))
    return keep, groups

def run():
    summary = {}
    allsplits = {}
    for tag, fn_ in [("mri", mri_samples), ("cxr", cxr_samples), ("gastro", gastro_samples)]:
        print(f"== {tag} ==", flush=True)
        classes, tr, va, te = fn_()
        allp = [s[0] for s in tr] + [s[0] for s in va] + [s[0] for s in te]
        lab = [s[1] for s in tr] + [s[1] for s in va] + [s[1] for s in te]
        H = hash_all(allp, tag)
        n_tr, n_va = len(tr), len(va)
        itr = np.arange(0, n_tr); iva = np.arange(n_tr, n_tr + n_va); ite = np.arange(n_tr + n_va, len(allp))
        dh, ph = H["dh"], H["ph"]
        # within-train exact (pixel) duplicates
        px_tr = H["px"][itr]
        _, cnt_px = np.unique(px_tr, return_counts=True)
        n_px_dup_tr = int((cnt_px - 1).clip(min=0).sum())
        # exact dup counts test-internal and test-vs-train
        px_te = H["px"][ite]
        px_train_set = set(px_tr.tolist())
        te_px_in_train = int(sum(1 for x in px_te if x in px_train_set))
        u, c = np.unique(px_te, return_counts=True)
        te_px_intra = int((c - 1).clip(min=0).sum())
        # within-train near-dup dedup
        keep_tr_local, groups = dedup_within(dh[itr], ph[itr], thr=4)
        keep_tr = itr[keep_tr_local]
        drop_tr = np.setdiff1d(itr, keep_tr)
        # cross-split distances to retained train
        d_va = min_dist({"dh": dh[iva], "ph": ph[iva]}, dh[keep_tr], ph[keep_tr])
        d_te = min_dist({"dh": dh[ite], "ph": ph[ite]}, dh[keep_tr], ph[keep_tr])
        clean_va = iva[d_va > 4]; clean_te = ite[d_te > 4]
        # distances for the ORIGINAL train (all) for audit reporting
        d_te_all = min_dist({"dh": dh[ite], "ph": ph[ite]}, dh[itr], ph[itr])
        d_va_all = min_dist({"dh": dh[iva], "ph": ph[iva]}, dh[itr], ph[itr])
        stats = {}
        for thr in (0, 2, 4, 8):
            stats[f"te_leak_H{thr}"] = int((d_te_all <= thr).sum())
            stats[f"va_leak_H{thr}"] = int((d_va_all <= thr).sum())
        summary[tag] = dict(classes=classes, n_train=len(tr), n_val=len(va), n_test=len(te),
                            n_px_dup_within_train=n_px_dup_tr,
                            n_test_px_in_train=te_px_in_train,
                            n_test_px_dup_internal=te_px_intra,
                            n_train_dropped_neardup=len(drop_tr),
                            n_train_clean=len(keep_tr), n_val_clean=len(clean_va), n_test_clean=len(clean_te),
                            leak=stats,
                            test_min_dist_hist={str(k): int((d_te_all == k).sum()) for k in range(0, 9)},
                            clean_test_class_counts=[int((np.array(lab)[clean_te] == c).sum()) for c in range(len(classes))],
                            clean_train_class_counts=[int((np.array(lab)[keep_tr] == c).sum()) for c in range(len(classes))])
        allsplits[tag] = dict(paths=allp, labels=lab,
                              train=[int(i) for i in keep_tr], val=[int(i) for i in clean_va],
                              test=[int(i) for i in clean_te],
                              train_dropped=[int(i) for i in drop_tr],
                              val_dropped=[int(i) for i in iva if i not in set(clean_va.tolist())],
                              test_dropped=[int(i) for i in ite if i not in set(clean_te.tolist())])
        print(json.dumps(summary[tag], indent=1), flush=True)

    # SMNR cohorts
    print("== smnr ==", flush=True)
    z, eso = zline_samples()
    Hz = hash_all(z, "zline")
    train_neg = []
    for folder in NORMAL: train_neg += sorted(glob.glob(os.path.join(GASTRO_ROOT, folder, "*.jpg")))
    Hn = hash_all(train_neg, "trainneg")
    He = hash_all(eso, "eso")
    # z-line clusters (near-dup groups) -> split groups in half
    keepz, grpz = dedup_within(Hz["dh"], Hz["ph"], thr=4)
    clusters = {}
    for rep, members in grpz.items(): clusters[rep] = [rep] + members
    for k in keepz: clusters.setdefault(k, [k])
    keys = sorted(clusters.keys())
    rng = np.random.RandomState(42); rng.shuffle(keys)
    a_idx, b_idx = [], []
    for n, k in enumerate(keys):
        (a_idx if n % 2 == 0 else b_idx).extend(clusters[k])
    a_idx = np.array(sorted(a_idx)); b_idx = np.array(sorted(b_idx))
    dz = min_dist(Hz, Hn["dh"], Hn["ph"]).astype(np.uint16)
    de = min_dist(He, Hn["dh"], Hn["ph"]).astype(np.uint16)
    cross = np.minimum(hamming(Hz["dh"][a_idx], Hz["dh"][b_idx]), hamming(Hz["ph"][a_idx], Hz["ph"][b_idx]))
    summary["smnr"] = dict(n_zline=len(z), n_eso=len(eso), n_train_neg=len(train_neg),
                           zline_vs_trainneg_min={f"H{t}": int((dz <= t).sum()) for t in (0, 2, 4, 8)},
                           eso_vs_trainneg_min={f"H{t}": int((de <= t).sum()) for t in (0, 2, 4, 8)},
                           n_clusters=len(clusters), smnr_train_neg=len(a_idx), control_heldout=len(b_idx),
                           cross_min_dist=int(cross.min()))
    print(json.dumps(summary["smnr"], indent=1), flush=True)
    allsplits["smnr"] = dict(zline=z, eso=eso, train_neg=train_neg,
                             smnr_neg=[z[i] for i in a_idx], control=[z[i] for i in b_idx])
    with open(os.path.join(OUT, "splits.json"), "w") as f: json.dump(allsplits, f)
    with open(os.path.join(OUT, "audit_summary.json"), "w") as f: json.dump(summary, f, indent=1)
    print("DONE")

if __name__ == "__main__":
    run()
