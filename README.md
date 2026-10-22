# MediScan AI

A tri-modality medical imaging diagnostic system that classifies brain MRI, chest
X-ray and endoscopy images with a shared dual-path CNN architecture, calibrated
decision thresholds and Grad-CAM explainability — served through a FastAPI
backend and a Next.js web interface.

> **Research and coursework project.** Not a medical device. Predictions must not
> be used for clinical decision-making.

---

## Table of contents

- [What it does](#what-it-does)
- [Highlights](#highlights)
- [Repository layout](#repository-layout)
- [Quick start](#quick-start)
  - [1. Backend](#1-backend)
  - [2. Frontend](#2-frontend)
  - [3. Single-image CLI](#3-single-image-cli)
- [API reference](#api-reference)
- [Datasets](#datasets)
- [Model weights](#model-weights)
- [Training and evaluation](#training-and-evaluation)
- [What is (and is not) in this repository](#what-is-and-is-not-in-this-repository)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What it does

Three independent heads, one shared architecture (`DualPathNet`): a dual-path
convolutional backbone feeding a refinement block, with a binary or 4-way
classification head per modality.

| Modality | Model | Classes | Accuracy | Notes |
| -------- | ----- | ------- | -------- | ----- |
| Brain MRI | `TumorNet` | Glioma, Meningioma, Pituitary adenoma, No tumor | 94.94 % | AUC 0.9874 |
| Chest X-ray | `PneumoNet` | Normal, Pneumonia | 90.06 % | Sensitivity 90.00 %, specificity 90.17 %, threshold 0.780 |
| Endoscopy | `GastroNet` | Normal mucosa, Mucosal ulcer | 96.06 % | Sensitivity 94.49 %, specificity 96.84 %, threshold 0.580 |

The binary modalities use thresholds calibrated on the validation split rather
than a naive 0.5 cut-off; they are stored inside each checkpoint.

## Highlights

- **Shared architecture** — `DualPathNet` is reused across all three tasks;
  `TumorNet` extends it with a 4-way head.
- **Explainability** — a live Grad-CAM engine returns the heatmap as a base64
  JPEG data URL, so the web client can display it without extra storage.
- **Automatic modality detection** — colour-variance and border heuristics pick
  brain / chest / gastro when the caller does not specify one.
- **Quantization** — `optimize_models.py` produces INT8 variants for edge
  deployment.
- **Measured latency** — `benchmark_latency.py` reports single-pass, 4-pass TTA
  and Grad-CAM timings separately.

## Repository layout

```
.
├── api_server.py                # FastAPI inference server + CLI diagnosis mode
├── train_tumornet.py            # Brain MRI training pipeline  (TumorNet)
├── train_pneumonet.py           # Chest X-ray training pipeline (DualPathNet)
├── train_gastronet.py           # Endoscopy training pipeline   (GastroNet)
├── evaluate_*.py                # Test-set metrics, ROC, confusion matrix, Grad-CAM samples
├── optimize_models.py           # INT8 quantization of the trained checkpoints
├── benchmark_latency.py         # Inference latency benchmark
│
├── checkpoints/                 # The three deployable *_best.pth weights
├── results/                     # Evaluation figures (ROC, curves, confusion matrices, Grad-CAM)
├── mediscan-ai/                 # Next.js 16 web application
│   ├── app/ components/ lib/ design-system/
│   ├── package.json
│   └── .env.example
│
├── requirements.txt
└── README.md
```

Supporting documents worth reading: `CNN_ARCHITECTURE_EXPLAINED.md` (model
internals), `CODE_WALKTHROUGH_GUIDE.md` (file-by-file tour),
`MODEL_OVERVIEW_SIMPLE.md` (plain-language summary),
`OPTIMIZING_PARAMETERS_README.md` (quantization) and `CHANGELOG.md`.

## Quick start

**Requirements:** Python 3.11+, Node.js 20+.

### 1. Backend

```bash
git clone https://github.com/jayeshgamer/MediScan.git
cd MediScan

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python api_server.py --serve
```

The server starts on <http://127.0.0.1:8000> with interactive docs at
`/docs`. If your machine has no CUDA, install the CPU wheels first:

```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

The three shipped checkpoints are picked up automatically from `checkpoints/`.
A modality whose checkpoint is missing is simply absent from `loaded_models` —
check `GET /health` after starting.

### 2. Frontend

```bash
cd mediscan-ai
npm install
npm run dev          # http://localhost:3000
```

The app defaults to `http://localhost:8000` for the API. To point it elsewhere,
copy `.env.example` to `.env.local` and set `NEXT_PUBLIC_API_URL`.

Production build:

```bash
npm run build && npm start
```

### 3. Single-image CLI

No server needed:

```bash
python api_server.py --image results/live_test_chest.jpg --save-heatmap out.jpg
python api_server.py --image results/live_test_brain.jpg --modality brain
```

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--image` | – | Image to diagnose (omit to start the server) |
| `--modality` | `auto` | `auto`, `brain`, `chest` or `gastro` |
| `--save-heatmap` | `gradcam_result.jpg` | Where to write the Grad-CAM overlay |
| `--serve` | off | Start the FastAPI server |
| `--host` / `--port` | `0.0.0.0` / `8000` | Server bind address |

## API reference

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET` | `/` , `/health` | Status, device, loaded models, calibrated thresholds |
| `GET` | `/modalities` | Supported modalities, class names and reported metrics |
| `POST` | `/predict` | Multipart upload (`image`, optional `modality`) → full report + Grad-CAM |

`POST /predict` returns the predicted class, per-class probabilities, calibrated
threshold decision, severity, findings, recommendations and a
`gradcam_base64` data URL.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "image=@results/live_test_chest.jpg" \
  -F "modality=chest"
```

## Datasets

Datasets are **not** in this repository (they total several GB). Place them at
the paths below before training or evaluating.

| Folder | Used by | Expected layout |
| ------ | ------- | --------------- |
| `Brain Tumor MRI Dataset/` | `TumorNet` | `Training/<Class>/…` and `Testing/<Class>/…` (ImageFolder) |
| `chest_xray/` | `PneumoNet` | `train/<Class>/…`, `val/<Class>/…`, `test/<Class>/…` |
| `labeled-images/` | `GastroNet` | `image-labels.csv` + `upper-gi-tract/`, `lower-gi-tract/` tree |

The endoscopy corpus is the HyperKvasir-style labelled image set (`image-labels.csv`
holds the `Video file, Organ, Finding, Classification` mapping). The brain and
chest corpora are the standard public Kaggle MRI and pneumonia radiograph
releases. Check each dataset's own licence before redistributing images.

Override the locations at runtime:

```bash
python train_tumornet.py   --data_dir "Brain Tumor MRI Dataset" --epochs 20 --batch_size 32
python train_pneumonet.py  --data_dir chest_xray
python train_gastronet.py  --data_dir labeled-images
```

## Model weights

`checkpoints/` ships the three files the API loads — `TumorNet_best.pth`,
`PneumoNet_best.pth`, `GastroNet_best.pth` (~57 MB each) — so a clone can run
inference immediately. Baseline, epoch-6 and INT8 variants stay local. See
[`checkpoints/README.md`](checkpoints/README.md) for the checkpoint format and
how to regenerate each file.

## Training and evaluation

All training scripts share the same flags: `--data_dir`, `--epochs`,
`--batch_size`, `--samples_per_epoch`, `--val_samples`, `--lr`, `--threads`,
`--save_dir`, `--results_dir`; `train_gastronet.py` adds `--resume`.

```bash
python train_tumornet.py --epochs 20           # -> checkpoints/TumorNet_best.pth
python evaluate_tumornet.py                    # -> results/*.png

python train_pneumonet.py --epochs 20          # -> checkpoints/PneumoNet_best.pth
python evaluate_pneumonet.py

python train_gastronet.py --epochs 20          # -> checkpoints/GastroNet_best.pth
python evaluate_gastronet.py

python optimize_models.py                      # INT8 variants
python benchmark_latency.py                    # latency table
```

Evaluation scripts accept `--checkpoint`, `--data_dir` and `--no_tta`, and write
confusion matrices, ROC curves, training curves and Grad-CAM samples into
`results/`.

## What is (and is not) in this repository

Nobody should have to clone 12 GB to read the code, so the following are
git-ignored and expected to exist only locally. Clone size with weights is
roughly 200 MB.

| Excluded | Size | Why |
| -------- | ---- | --- |
| `.venv/` | 4.8 GB | Recreate with `pip install -r requirements.txt` |
| `mediscan-ai/node_modules/`, `.next/` | 0.6 GB | Recreate with `npm install` |
| `labeled-images/`, `chest_xray/`, `Brain Tumor MRI Dataset/` | 6.3 GB | Public datasets; bring your own copy |
| `*_baseline.pth`, `*_best_epoch6.pth`, `*_quantized.pth` | 400 MB | Experiment artifacts; regenerable |
| `scratch/`, `.agents/` | small | Local scratch and tooling |
| `.env.local` | – | Local config — use `.env.example` |

Everything above stays on disk; deleting these rules locally will make them
visible to Git again.

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `loaded_models` is empty in `/health` | Checkpoints missing — see [Model weights](#model-weights) |
| Frontend gets `ECONNREFUSED` | Backend not running, or `NEXT_PUBLIC_API_URL` points at the wrong port |
| `RuntimeError: no trained model for modality` | That modality's `*_best.pth` is absent |
| Torch install is huge / CUDA mismatch | Use the CPU index URL in [Backend](#1-backend) |

## License

No licence file is included yet. This is coursework; add a licence (for example
MIT, or a restrictive one) before distributing or publishing derivatives.
