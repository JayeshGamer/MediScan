
"""Evaluate frozen GastroNet on the held-out Z-line control (same subset as SMNR) for a fair pre/post comparison."""
import os, sys, json, numpy as np, torch
ROOT = r"D:\CapStone-2"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "clean_study"))
from retrain import tta_probs, eval_bin, DS, EVAL_TF, load_ckpt

def main():
    sp = json.load(open(os.path.join(ROOT, "clean_study", "clean_splits.json")))["smnr"]
    z = json.load(open(os.path.join(ROOT, "clean_study", "splits.json")))["smnr"]["zline"]
    control = sp["control"]; eso = sp["eso"]
    model, name = load_ckpt("dual", 2, os.path.join(ROOT, "checkpoints", "GastroNet_best.pth"))
    out = {}
    yz = np.array([0] * len(z)); Pz = tta_probs(model, DS(z, yz, EVAL_TF))
    out["zline_full"] = eval_bin(yz, Pz[:, 1], 0.5)
    yc = np.array([0] * len(control) + [1] * len(eso)); Pc = tta_probs(model, DS(control + eso, yc, EVAL_TF))
    out["control_heldout"] = eval_bin(yc, Pc[:, 1], 0.5)
    os.makedirs(os.path.join(ROOT, "clean_study", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "clean_study", "results", "frozen_control.json"), "w"), indent=1)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("accuracy", "sensitivity", "specificity", "f1", "n")} for k, v in out.items()}, indent=1))

if __name__ == "__main__":
    main()
