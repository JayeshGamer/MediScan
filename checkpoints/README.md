# Model checkpoints

`api_server.py` loads three files from this folder:

| File                  | Model      | Task                                            |
| --------------------- | ---------- | ----------------------------------------------- |
| `TumorNet_best.pth`   | TumorNet   | Brain MRI — Glioma / Meningioma / Pituitary / No Tumor |
| `PneumoNet_best.pth`  | PneumoNet  | Chest X-ray — Normal / Pneumonia                |
| `GastroNet_best.pth`  | GastroNet  | Endoscopy — Normal mucosa / Mucosal ulcer       |

Each checkpoint is a dictionary saved by the `train_*.py` scripts:

```python
{
    "model_state":     ...,   # state_dict for the model class
    "num_classes":     4,     # or 2
    "best_threshold":  0.78,  # calibrated decision threshold (binary heads)
}
```

## What is tracked

Only the three `*_best.pth` files above are committed (~57 MB each), so that a
fresh clone can run inference without retraining. Large weights in Git are not
ideal — if you prefer a lean repository, delete them from the index and publish
them as a release asset instead:

```bash
git rm --cached checkpoints/*.pth
# upload the files to a GitHub Release, then update the README instructions
```

## What is not tracked

The following stay local (see `.gitignore`):

- `*_baseline.pth` — untrained/reference runs
- `*_best_epoch6.pth` — intermediate snapshot
- `*_quantized.pth` — INT8 quantized variants produced by `optimize_models.py`

Regenerate them at any time:

```bash
python optimize_models.py            # writes *_quantized.pth here
python train_tumornet.py             # writes TumorNet_best.pth + results
python train_pneumonet.py
python train_gastronet.py
```
