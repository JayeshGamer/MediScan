
import subprocess, sys, os, json
PY = os.path.join(r"D:\CapStone-2", ".venv", "Scripts", "python.exe")
ROOT = r"D:\CapStone-2"
R = os.path.join(ROOT, "clean_study", "retrain.py")
QUEUE = [
    ("evalonly", ["--task", "evalonly"]),
    ("mri_dual", ["--task", "mri", "--arch", "dual"]),
    ("cxr_dual", ["--task", "cxr", "--arch", "dual"]),
    ("gastro_dual", ["--task", "gastro", "--arch", "dual"]),
    ("mri_r18", ["--task", "mri", "--arch", "r18"]),
    ("cxr_r18", ["--task", "cxr", "--arch", "r18"]),
    ("gastro_r18", ["--task", "gastro", "--arch", "r18"]),
    ("mri_r50", ["--task", "mri", "--arch", "r50"]),
    ("cxr_r50", ["--task", "cxr", "--arch", "r50"]),
    ("gastro_r50", ["--task", "gastro", "--arch", "r50"]),
    ("smnr_dual", ["--task", "smnr", "--arch", "dual"]),
]
only = sys.argv[1] if len(sys.argv) > 1 else None
extra = sys.argv[2:] if len(sys.argv) > 2 else []
for tag, args in QUEUE:
    if only and only != "all" and only != tag: continue
    print("=" * 20, tag, "=" * 20, flush=True)
    rc = subprocess.call([PY, R] + args + extra, cwd=ROOT)
    print("rc", rc, flush=True)
print("ALL_DONE")
