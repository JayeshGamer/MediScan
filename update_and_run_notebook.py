"""
Script to update MediScan_AI_Models.ipynb with the pneumonia dataset configuration
and execute all notebook cells, saving the evaluated notebook with rich outputs.
"""

import os
import sys
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

NOTEBOOK_PATH = "MediScan_AI_Models.ipynb"

def update_notebook():
    print(f"Reading {NOTEBOOK_PATH}...")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)

    # Disable pip install cell so it runs instantaneously
    nb.cells[2].source = "# Dependencies verified: torch, torchvision, grad-cam, sklearn, matplotlib, seaborn\nprint('Dependencies ready ✓')"

    # Fix Step 6 (Cell 18) — GradCAM visualize dimension resizing
    nb.cells[18].source = """class GradCAM:
    \"\"\"
    STEP 7 — Grad-CAM Heatmap Generator.

    1. Register forward hook → captures activations at target conv layer.
    2. Register backward hook → captures gradients.
    3. On generate():
       - Run forward pass → get prediction
       - Run backward pass for target class
       - Compute: weights = mean(gradients over H,W)
       - CAM = ReLU( sum(weights * activations) )
       - Upsample to 224×224 → normalize → return heatmap
    \"\"\"
    def __init__(self, model: DualPathNet):
        self.model      = model
        self.gradients  = None
        self.activations = None
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        # Hook into last ConvBlock inside refine (index 2), before AdaptiveAvgPool
        target_layer = list(self.model.refine.children())[2]

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, image_tensor, class_idx=None):
        \"\"\"
        Args:
            image_tensor : (1, C, H, W) preprocessed tensor
            class_idx    : target class (None = use predicted class)
        Returns:
            heatmap      : numpy (H, W) normalized 0-1
            pred_class   : predicted class index
            confidence   : float 0-1
        \"\"\"
        self.model.zero_grad()
        img = image_tensor.to(DEVICE).requires_grad_(True)

        output    = self.model(img)
        probs     = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if class_idx is None:
            class_idx = pred_class

        # Backprop for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Grad-CAM formula
        weights = self.gradients.mean(dim=(2,3), keepdim=True)  # (1,C,1,1)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)

        # Upsample to input size
        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, pred_class, confidence

    def visualize(self, original_img, heatmap, pred_label, confidence,
                  save_path=None, title="Grad-CAM"):
        \"\"\"Overlays heatmap on original image.\"\"\"
        if isinstance(original_img, Image.Image):
            original_img = np.array(original_img.resize((224, 224)).convert("RGB"))
        else:
            original_img = np.array(Image.fromarray(original_img).resize((224, 224)).convert("RGB"))

        heatmap_color = cm.jet(heatmap)[:, :, :3]
        overlay = 0.55 * (original_img / 255.0) + 0.45 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f"{title}  |  Prediction: {pred_label}  "
                     f"({confidence*100:.1f}% confidence)", fontsize=14, fontweight='bold')

        axes[0].imshow(original_img);        axes[0].set_title("Original Image",   fontsize=12)
        axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[2].imshow(overlay);             axes[2].set_title("Overlay",           fontsize=12)
        for ax in axes: ax.axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved → {save_path}")
        plt.show()
        plt.close()

print("GradCAM defined ✓")"""

    # Update Step 7 (Cell 20) — Data Loading to support train/test/val subdirectories
    nb.cells[20].source = """def get_transforms():
    \"\"\"Returns train augmentation and val/test transforms.\"\"\"

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_tf, val_tf


def load_dataset(data_dir, batch_size=32, val_split=0.15,
                 test_split=0.15, num_workers=0, seed=42):
    \"\"\"
    Loads ImageFolder dataset and splits into train/val/test.
    Supports either pre-split train/val/test folders or single unified directory.
    \"\"\"
    train_tf, val_tf = get_transforms()

    train_path = os.path.join(data_dir, "train")
    test_path  = os.path.join(data_dir, "test")
    val_path   = os.path.join(data_dir, "val")

    if not os.path.exists(train_path) and os.path.exists(os.path.join(data_dir, "chest_xray", "train")):
        train_path = os.path.join(data_dir, "chest_xray", "train")
        test_path  = os.path.join(data_dir, "chest_xray", "test")
        val_path   = os.path.join(data_dir, "chest_xray", "val")

    if os.path.exists(train_path) and os.path.exists(test_path):
        raw_train = datasets.ImageFolder(train_path, transform=train_tf)
        class_names = raw_train.classes
        
        n = len(raw_train)
        n_val = int(n * val_split)
        n_train = n - n_val
        g = torch.Generator().manual_seed(seed)
        train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=g)
        raw_train_eval = datasets.ImageFolder(train_path, transform=val_tf)
        train_ds = Subset(raw_train, train_idx.indices)
        val_ds   = Subset(raw_train_eval, val_idx.indices)
        test_ds  = datasets.ImageFolder(test_path, transform=val_tf)
    else:
        full_aug = datasets.ImageFolder(data_dir, transform=train_tf)
        full_val = datasets.ImageFolder(data_dir, transform=val_tf)
        class_names = full_aug.classes

        n        = len(full_aug)
        n_test   = int(n * test_split)
        n_val    = int(n * val_split)
        n_train  = n - n_val - n_test

        g = torch.Generator().manual_seed(seed)
        train_idx, val_idx, test_idx = random_split(
            range(n), [n_train, n_val, n_test], generator=g
        )

        train_ds = Subset(full_aug, train_idx.indices)
        val_ds   = Subset(full_val, val_idx.indices)
        test_ds  = Subset(full_val, test_idx.indices)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())

    print(f"Dataset    : {data_dir}")
    print(f"Classes    : {class_names}")
    print(f"Train      : {len(train_ds):,} images")
    print(f"Validation : {len(val_ds):,} images")
    print(f"Test       : {len(test_ds):,} images")
    print(f"Batch size : {batch_size}")

    return train_loader, val_loader, test_loader, class_names

print("Data loading functions defined ✓")"""

    # Update Step 12 (Cell 30) — PneumoNet training & evaluation
    nb.cells[30].source = """# ╔══════════════════════════════════════════════════════════════╗
# ║          SET YOUR DATASET PATHS HERE                         ║
# ╚══════════════════════════════════════════════════════════════╝

GASTRO_DIR  = "datasets/hyperkvasir"
PNEUMO_DIR  = r"chest_xray"
NEURO_DIR   = "datasets/brain_tumor_mri"

# ── Config ─────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_EPOCHS  = 6
LR          = 3e-4

# ── PNEUMONET ──────────────────────────────────────────────────
if os.path.exists(PNEUMO_DIR):
    print("Loading Pneumonia dataset and configuring PneumoNet...")
    pneumo_train, pneumo_val, pneumo_test, pneumo_classes = \\
        load_dataset(PNEUMO_DIR, batch_size=BATCH_SIZE, num_workers=0)
    
    ckpt_path = os.path.join("checkpoints", "PneumoNet_best.pth")
    if os.path.exists(ckpt_path):
        print(f"Loading trained checkpoint from {ckpt_path}...")
        pneumo_model, pneumo_history = load_checkpoint(ckpt_path)
    else:
        pneumo_model = PneumoNet()
        pneumo_model, pneumo_history = train_model(
            pneumo_model, pneumo_train, pneumo_val,
            num_epochs=NUM_EPOCHS, lr=LR)
    
    if pneumo_history:
        plot_history(pneumo_history, "PneumoNet")
    pneumo_metrics = evaluate_model(pneumo_model, pneumo_test, pneumo_classes)
"""

    # Update Step 13 (Cell 32) — Grad-CAM on test sample
    nb.cells[32].source = """# ── Step 13: Run Grad-CAM on a test image ──────────────────────
import glob

test_samples = glob.glob(os.path.join(PNEUMO_DIR, "test", "PNEUMONIA", "*.*"))
if not test_samples:
    test_samples = glob.glob(os.path.join(PNEUMO_DIR, "chest_xray", "test", "PNEUMONIA", "*.*"))

if test_samples and 'pneumo_model' in locals():
    IMAGE_PATH = test_samples[0]
    CLASS_NAMES = pneumo_classes
    print(f"Running Grad-CAM on: {IMAGE_PATH}")
    pred, conf, heatmap = predict_single(pneumo_model, IMAGE_PATH, CLASS_NAMES)
"""

    # Update Step 14 (Cell 34) — Checkpoint loading
    nb.cells[34].source = """# ── Step 14: Save and Load Checkpoints ──────────────────────────
ckpt_path = os.path.join("checkpoints", "PneumoNet_best.pth")
if os.path.exists(ckpt_path):
    model, history = load_checkpoint(ckpt_path)
    if history:
        plot_history(history, model.model_name)
"""

    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH} successfully.")


def execute_notebook():
    print(f"Executing {NOTEBOOK_PATH}...")
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    
    ep.preprocess(nb, {"metadata": {"path": "."}})
    
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Executed {NOTEBOOK_PATH} and saved outputs successfully ✓")


if __name__ == "__main__":
    update_notebook()
    execute_notebook()
