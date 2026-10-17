
"""Content-cropped 256-bit perceptual descriptors + definitive clean-split construction."""
import os, json, hashlib
import numpy as np
from PIL import Image

ROOT = r"D:\CapStone-2"
OUT  = os.path.join(ROOT, "clean_study")
POP  = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

N = 32                      # 32x32 grid -> dHash 32*31 = 992 bits, pHash 32x32 = 1024 bits
W = 16                      # 16 uint64 words
def _words(bits):
    b = np.asarray(bits, dtype=np.uint8).ravel()
    v = np.zeros(W, dtype=np.uint64)
    for i in range(W):
        seg = b[i * 64:(i + 1) * 64]
        acc = 0
        for x in seg: acc = (acc << 1) | int(x)
        v[i] = acc
    return v

def content_crop(im, thresh=0.06, margin=0.02):
    g = np.asarray(im.convert("L").resize((256, 256), Image.BILINEAR), dtype=np.float32) / 255.0
    m = g > (g.max() * thresh if g.max() > 0 else 0.0)
    if m.sum() < 64: return im
    ys, xs = np.where(m)
    h, w = g.shape
    y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
    my = int(margin * (y1 - y0 + 1)); mx = int(margin * (x1 - x0 + 1))
    y0 = max(0, y0 - my); x0 = max(0, x0 - mx); y1 = min(h - 1, y1 + my); x1 = min(w - 1, x1 + mx)
    if (y1 - y0) < 8 or (x1 - x0) < 8: return im
    W0, H0 = im.size
    return im.crop((int(x0 / w * W0), int(y0 / h * H0), int((x1 + 1) / w * W0), int((y1 + 1) / h * H0)))

_D = np.cos(np.pi * (2 * np.arange(N)[:, None] + 1) * np.arange(N)[None, :] / (2.0 * N)) * np.sqrt(2.0 / N)
_D[0, :] *= np.sqrt(0.5)

def descriptors(p):
    with Image.open(p) as im0:
        im = im0.convert("RGB")
        c = content_crop(im)
        g = np.asarray(c.convert("L").resize((N + 1, N), Image.BILINEAR), dtype=np.int16)
        dh = _words((g[:, 1:] > g[:, :-1]).astype(np.uint8))
        a = np.asarray(c.convert("L").resize((N, N), Image.BILINEAR), dtype=np.float64)
    co = _D @ a @ _D.T
    low = co.ravel()
    med = np.median(low[1:])
    ph = _words((low > med).astype(np.uint8))
    return dh, ph

def hash_all(paths, tag):
    fn = os.path.join(OUT, f"hash2_{tag}.npz")
    if os.path.exists(fn):
        d = np.load(fn, allow_pickle=True)
        if len(d["paths"]) == len(paths) and list(d["paths"]) == list(paths):
            print(f"  [cache] {tag} {len(paths)}", flush=True); return d
    dh = np.zeros((len(paths), W), dtype=np.uint64); ph = np.zeros((len(paths), W), dtype=np.uint64)
    for i, p in enumerate(paths):
        dh[i], ph[i] = descriptors(p)
        if (i + 1) % 1000 == 0: print(f"  {tag} {i+1}/{len(paths)}", flush=True)
    d = dict(paths=np.array(paths), dh=dh, ph=ph)
    np.savez(fn, **d); return d

def hamming(A, B, ch=128):
    A = np.ascontiguousarray(A, dtype=np.uint64); B = np.ascontiguousarray(B, dtype=np.uint64)
    out = np.zeros((A.shape[0], B.shape[0]), dtype=np.uint16)
    nbytes = W * 8
    for i in range(0, A.shape[0], ch):
        X = np.bitwise_xor(A[i:i+ch, None, :], B[None, :, :])
        X = np.ascontiguousarray(X, dtype=np.uint64).view(np.uint8).reshape(X.shape[0], X.shape[1], nbytes)
        out[i:i+ch] = POP[X].sum(2)
    return out

def dedup(dh, ph, thr):
    rep_dh, rep_ph, rep_idx, groups = [], [], [], {}
    order = np.arange(len(dh))
    for i in order:
        if rep_idx:
            d = np.minimum(hamming(dh[i:i+1], np.array(rep_dh))[0], hamming(ph[i:i+1], np.array(rep_ph))[0])
            j = int(np.argmin(d))
            if d[j] <= thr:
                groups.setdefault(rep_idx[j], []).append(int(i)); continue
        rep_dh.append(dh[i]); rep_ph.append(ph[i]); rep_idx.append(int(i))
    return rep_idx, groups

def run():
    import audit as A
    summ = {}
    clean = {}
    for tag, fnb in (("mri", A.mri_samples), ("cxr", A.cxr_samples), ("gastro", A.gastro_samples)):
        classes, tr, va, te = fnb()
        paths = [s[0] for s in tr] + [s[0] for s in va] + [s[0] for s in te]
        lab = np.array([s[1] for s in tr] + [s[1] for s in va] + [s[1] for s in te])
        va = np.arange(len(tr), len(tr) + len(va)); te = np.arange(len(tr) + len(va), len(paths))
        print(f"== {tag} ==", flush=True)
        H = hash_all(paths, tag)
        dh, ph = H["dh"], H["ph"]
        orig_tr = np.arange(len(tr))
        keep_idx, groups = dedup(dh[orig_tr], ph[orig_tr], 12)
        tr_local = orig_tr[np.array(keep_idx, dtype=int)]
        rep_dh, rep_ph = dh[tr_local], ph[tr_local]
        d_va = np.minimum(hamming(dh[va], rep_dh), hamming(ph[va], rep_ph)).min(1)
        d_te = np.minimum(hamming(dh[te], rep_dh), hamming(ph[te], rep_ph)).min(1)
        d_te_all = np.minimum(hamming(dh[te], dh[orig_tr]), hamming(ph[te], ph[orig_tr])).min(1)
        hist = {str(k): int((d_te_all <= k).sum()) for k in range(0, 41, 2)}
        summ[tag] = dict(n_train_pool=int(len(orig_tr)), n_train_clean=int(len(tr_local)),
                         n_val=int(len(va)), n_test=int(len(te)),
                         train_dedup_dropped=int(len(orig_tr) - len(tr_local)),
                         test_leak_hist=hist,
                         leak_frac={f"T{t}": round(float((d_te_all <= t).mean()), 4) for t in (2, 6, 12, 20, 40)},
                         clean_test_n=int((d_te > 12).sum()), clean_val_n=int((d_va > 12).sum()),
                         clean_test_classes=[int((lab[te][d_te > 12] == c).sum()) for c in range(lab.max() + 1)],
                         orig_test_classes=[int((lab[te] == c).sum()) for c in range(lab.max() + 1)],
                         clean_train_classes=[int((lab[tr_local] == c).sum()) for c in range(lab.max() + 1)])
        print(json.dumps(summ[tag], indent=1), flush=True)
        clean[tag] = dict(paths=[paths[i] for i in tr_local],
                          labels=[int(lab[i]) for i in tr_local],
                          val=[paths[i] for i in va[d_va > 12]], val_labels=[int(lab[i]) for i in va[d_va > 12]],
                          test=[paths[i] for i in te[d_te > 12]], test_labels=[int(lab[i]) for i in te[d_te > 12]],
                          test_all=[paths[i] for i in te], test_all_labels=[int(lab[i]) for i in te])
    # ---- SMNR cohorts
    print("== smnr ==", flush=True)
    Sm = json.load(open(os.path.join(OUT, "splits.json")))["smnr"]
    z, eso, neg = Sm["zline"], Sm["eso"], Sm["train_neg"]
    Hz = hash_all(z, "zline"); He = hash_all(eso, "eso"); Hn = hash_all(neg, "trainneg")
    dz = np.minimum(hamming(Hz["dh"], Hn["dh"]), hamming(Hz["ph"], Hn["ph"])).min(1)
    de = np.minimum(hamming(He["dh"], Hn["dh"]), hamming(He["ph"], Hn["ph"])).min(1)
    keepz, grpz = dedup(Hz["dh"], Hz["ph"], 12)
    clusters = {k: [k] for k in keepz}
    for rep, mem in grpz.items(): clusters[rep] = [rep] + mem
    keys = sorted(clusters); rng = np.random.RandomState(42); rng.shuffle(keys)
    A, B = [], []
    for n, k in enumerate(keys): (A if n % 2 == 0 else B).extend(clusters[k])
    A = np.array(sorted(A)); B = np.array(sorted(B))
    crossAB = np.minimum(hamming(Hz["dh"][A], Hz["dh"][B]), hamming(Hz["ph"][A], Hz["ph"][B]))
    crossZA = np.minimum(hamming(Hz["dh"], Hz["dh"][A]), hamming(Hz["ph"], Hz["ph"][A]))
    Bset = set(int(x) for x in B)
    n_control_clusters = len([c for c in clusters if int(c) in Bset])
    summ["smnr"] = dict(n_zline=len(z), n_eso=len(eso), n_train_neg=len(neg), n_clusters=len(clusters),
                        smnr_neg=len(A), control=len(B), n_control_clusters=n_control_clusters,
                        zline_vs_trainneg={f"T{t}": int((dz <= t).sum()) for t in (2, 6, 12, 20)},
                        eso_vs_trainneg={f"T{t}": int((de <= t).sum()) for t in (2, 6, 12, 20)},
                        cross_A_B_min=int(crossAB.min()), cross_A_B_le12=int((crossAB <= 12).sum()),
                        dz_hist={str(k): int((dz == k).sum()) for k in range(0, 13)})
    print(json.dumps(summ["smnr"], indent=1), flush=True)
    clean["smnr"] = dict(train_neg=neg, smnr_neg=[z[i] for i in A], control=[z[i] for i in B], eso=eso)
    json.dump(summ, open(os.path.join(OUT, "summary2.json"), "w"), indent=1)
    json.dump(clean, open(os.path.join(OUT, "clean_splits.json"), "w"))
    print("DONE2")

if __name__ == "__main__":
    run()
