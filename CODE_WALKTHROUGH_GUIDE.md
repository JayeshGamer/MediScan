# MediScan AI — Code Walkthrough & Line-by-Line Navigation Map
**How to Find and Explain Every Function, Layer, and Preprocessing Step to Your Mentor**

---

## 🎯 Emergency Quick Reference: "Where Is Preprocessing?"

If your mentor asks: ***"Show me the section that does preprocessing!"***

Open any of the training files and go straight to the `get_transforms()` function:

| Script | Line Numbers | Function Name |
| :--- | :--- | :--- |
| **`train_pneumonet.py`** | **Lines 315 – 336** | `def get_transforms():` |
| **`train_tumornet.py`** | **Lines 300 – 321** | `def get_transforms():` |
| **`train_gastronet.py`** | **Lines 361 – 382** | `def get_transforms():` |
| **`MediScan_AI_Models.ipynb`** | **Cell 20** | `def get_transforms():` |

---

## 1. Line-by-Line Explanation of the Preprocessing Code

Here is the exact code from `train_pneumonet.py` (lines 315–336):

```python
def get_transforms():
    """High-yield data augmentation for chest radiography."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=12),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), value="random")
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_tf, eval_tf
```

### What Each Line Does (Explain This to Your Mentor):

1. **`transforms.Resize((224, 224))`**:
   - *Why*: Medical scans arrive in wild resolutions (e.g., $2000 \times 1800$ for chest X-rays, $512 \times 512$ for MRIs). Convolutional neural networks require a uniform square tensor dimension. $224 \times 224$ is the optimal clinical standard balancing diagnostic detail and GPU memory.
2. **`transforms.RandomHorizontalFlip(p=0.5)`**:
   - *Why*: A 50% chance of mirroring the image horizontally. This doubles effective dataset diversity and teaches the model that tumors and ulcers can appear on either side without altering pathology.
3. **`transforms.RandomRotation(degrees=12)`**:
   - *Why*: Patients tilt slightly during MRI or X-ray acquisition. Rotating by $\pm 12^\circ$ simulates slight patient posture variations.
4. **`transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06))`**:
   - *Why*: Translates (shifts) the scan by up to 6% horizontally/vertically and zooms by up to 6%. This ensures the network doesn't overfit to an exact lesion centering.
5. **`transforms.ColorJitter(brightness=0.15, contrast=0.15)`**:
   - *Why*: Different imaging machines and hospitals have different sensor calibrations, lighting, and exposure settings. Jittering brightness and contrast by 15% makes the model invariant to scanner exposure.
6. **`transforms.ToTensor()`**:
   - *Why*: Converts raw PIL images (with integer pixel values $0 - 255$) into a PyTorch floating-point tensor scaled to $[0.0, 1.0]$ with channel dimension first (`[C, H, W]`).
7. **`transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`**:
   - *Why*: Z-score standardizes the pixel values:
     $$\text{pixel}_{\text{norm}} = \frac{\text{pixel} - \text{mean}}{\text{std}}$$
     This centers data around zero, preventing gradient explosions and allowing stable backpropagation.
8. **`transforms.RandomErasing(p=0.15, scale=(0.02, 0.15), value="random")`**:
   - *Why*: Randomly blanks out a small patch ($2\% - 15\%$ area) with noise. This forces the model to look at multiple surrounding anatomical cues rather than memorizing a single localized artifact.
9. **`eval_tf` (Why is it different?)**:
   - *Crucial Mentor Note*: Notice that `eval_tf` has **NO random augmentations** (no flips, no rotations, no jitter, no erasing). In validation and testing, we evaluate the patient scan in its true, unaltered state with only `Resize`, `ToTensor`, and `Normalize`.

---

## 2. Complete Code Map by Topic

Use this directory map to answer any question about where specific logic is implemented in the repository:

```
CapStone-2/
├── train_pneumonet.py      <-- PneumoNet training, dataloading, loss, threshold calibration
├── train_tumornet.py       <-- TumorNet multi-class training and SWA logic
├── train_gastronet.py      <-- GastroNet endoscopy sample pairing & training
├── evaluate_pneumonet.py   <-- Clinical test metrics, confusion matrix, Grad-CAM
├── evaluate_tumornet.py    <-- 4-class evaluation, multi-class ROC, Grad-CAM
├── evaluate_gastronet.py   <-- Endoscopy test metrics, ROC curve, Grad-CAM
├── optimize_models.py      <-- PyTorch Dynamic INT8 Quantization & CPU latency benchmarking
```

---

### Topic 1: Dataset Loading & Class Balancing
- **Where**:
  - `train_pneumonet.py`: Lines 339–425 (`build_dataloaders`)
  - `train_tumornet.py`: Lines 343–401 (`build_dataloaders`)
  - `train_gastronet.py`: Lines 385–480 (`collect_samples` and `build_dataloaders`)
- **Key Code to Show Mentor**:
  ```python
  # Inverse-frequency weighted sampling to eliminate class imbalance:
  train_targets = [s[1] for s in train_samples]
  class_counts = np.bincount(train_targets)
  class_weights = 1.0 / class_counts.astype(np.float32)
  sample_weights = [class_weights[t] for t in train_targets]

  sampler = WeightedRandomSampler(
      weights=sample_weights,
      num_samples=len(train_dataset),
      replacement=True
  )
  ```

---

### Topic 2: Neural Network Architecture Definition
- **Where**:
  - `train_pneumonet.py`: Lines 31–240
  - `train_tumornet.py`: Lines 31–240
- **Components to Point Out**:
  - `ResidualConvBlock`: Lines 31–75 (Two Conv layers + BatchNorm + SiLU + Skip Shortcut)
  - `ChannelAttention`: Lines 80–110 (Channel pooling + 2-layer MLP + Sigmoid)
  - `SpatialAttention`: Lines 112–135 (Spatial pooling + $7 \times 7$ Conv + Sigmoid)
  - `CBAM`: Lines 137–152 (Sequential channel + spatial attention application)
  - `DualPathNet` / `PneumoNet`: Lines 155–240 (Parallel Path A + Path B, `join_conv`, Dual Pooling, Classifier MLP)

---

### Topic 3: Loss Function Definition (Focal Loss)
- **Where**:
  - `train_pneumonet.py`: Lines 243–275 (`ClassWeightedFocalLoss`)
  - `train_tumornet.py`: Lines 243–275 (`MultiClassFocalLoss`)
- **Key Code to Show Mentor**:
  ```python
  class ClassWeightedFocalLoss(nn.Module):
      def __init__(self, gamma=2.0, label_smoothing=0.04):
          super().__init__()
          self.gamma = gamma
          self.smoothing = label_smoothing

      def forward(self, inputs, targets):
          ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.smoothing)
          pt = torch.exp(-ce_loss)
          focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
          return focal_loss.mean()
  ```

---

### Topic 4: Learning Rate Scheduling (Warmup + Cosine Annealing)
- **Where**:
  - `train_pneumonet.py`: Lines 473–479 (`get_warmup_cosine_scheduler`)
  - `train_tumornet.py`: Lines 436–442 (`get_warmup_cosine_scheduler`)
- **Key Code to Show Mentor**:
  ```python
  def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
      def lr_lambda(epoch):
          if epoch < warmup_epochs:
              return float(epoch + 1) / float(max(1, warmup_epochs))
          progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
          return max(min_lr / base_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
      return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
  ```

---

### Topic 5: Training Loop & Mixed Precision (AMP)
- **Where**:
  - `train_pneumonet.py`: Lines 504–594 (in `train_pneumonet`)
  - `train_tumornet.py`: Lines 475–560 (in `train_tumornet`)
- **Key Code to Show Mentor**:
  ```python
  scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

  # Inside the batch loop:
  with torch.amp.autocast('cuda', enabled=use_amp):
      outputs = model(imgs)
      loss = criterion(outputs, targets)

  scaler.scale(loss).backward()
  scaler.unscale_(optimizer)
  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  scaler.step(optimizer)
  scaler.update()
  ```

---

### Topic 6: Clinical Decision Threshold Calibration
- **Where**:
  - `train_pneumonet.py`: Lines 615–631
  - `train_gastronet.py`: Lines 656–672
- **Key Code to Show Mentor**:
  ```python
  # Grid-search across 31 candidate thresholds on validation set to maximize Youden's J:
  for thresh in np.linspace(0.35, 0.65, 31):
      preds_t = (all_val_probs_np >= thresh).astype(int)
      cm = confusion_matrix(all_val_targets_np, preds_t)
      if cm.shape == (2, 2):
          tn, fp, fn, tp = cm.ravel()
          sens = tp / (tp + fn)
          spec = tn / (tn + fp)
          j_stat = sens + spec - 1.0
          if j_stat > best_j:
              best_j = j_stat
              best_threshold = float(thresh)
  ```

---

### Topic 7: Model Explainability (Grad-CAM Saliency Maps)
- **Where**:
  - `evaluate_pneumonet.py`: Lines 35–90 (`GradCAM` class) and Lines 290–315 (`generate_and_save_gradcam`)
  - `evaluate_tumornet.py`: Lines 35–90 and Lines 290–315
- **How to Explain the Hooks**:
  - The model hooks into the **last convolutional block** (`model.refine`) before the dual pooling layer.
  - A forward hook captures the 2D feature map activations: $A^k$.
  - A backward hook captures the gradient of the predicted class with respect to those activations: $\frac{\partial Y^c}{\partial A^k}$.
  - The gradients are global-average-pooled into importance weights $\alpha_k$, producing a heatmap:
    $$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_k \alpha_k A^k\right)$$
  - The heatmap is upsampled and overlaid onto the original medical image in bright red/yellow to highlight the pathology.

---

### Topic 8: INT8 Dynamic Quantization
- **Where**:
  - `optimize_models.py`: Lines 58–68 (`quantize_model`)
- **Key Code to Show Mentor**:
  ```python
  def quantize_model(model):
      quantized_model = torch.ao.quantization.quantize_dynamic(
          model,
          {nn.Linear},
          dtype=torch.qint8
      )
      return quantized_model
  ```
  - Mentions that this quantizes the weights of fully connected projection layers from 32-bit floats to 8-bit integers, reducing memory footprint by ~3.5% and optimizing CPU throughput without retraining.
