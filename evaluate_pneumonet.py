"""
MediScan AI — PneumoNet Evaluation & Grad-CAM Pipeline
Evaluates best checkpoint on Kaggle Chest X-Ray test split,
computes clinical classification metrics, and generates Grad-CAM visualizations.
"""

import os
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# Import model architecture from training script
from train_pneumonet import DualPathNet, PneumoNet, get_transforms


# ==============================================================================
# Grad-CAM Engine
# ==============================================================================

class GradCAM:
    """
    Grad-CAM Heatmap Generator for PneumoNet:
    Hooks into the final convolutional feature extractor in the refinement block.
    """
    def __init__(self, model: DualPathNet, device: torch.device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        # Target layer: last ConvBlock inside refine before AdaptiveAvgPool2d
        target_layer = list(self.model.refine.children())[2]

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, image_tensor, class_idx=None):
        """
        Generates Grad-CAM heatmap for a given input tensor.
        """
        self.model.zero_grad()
        img = image_tensor.to(self.device).requires_grad_(True)

        output = self.model(img)
        probs = F.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        confidence = probs[0, pred_class].item()

        if class_idx is None:
            class_idx = pred_class

        # Backpropagation for target class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)

        # Global average pooling of gradients as channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to 224x224
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam, pred_class, confidence

    def save_visualization(self, original_img_pil, heatmap, pred_label, true_label, confidence, save_path):
        """Creates and saves a 3-panel figure: Original, Heatmap, Overlay."""
        img_resized = original_img_pil.resize((224, 224)).convert("RGB")
        img_np = np.array(img_resized)

        heatmap_color = cm.jet(heatmap)[:, :, :3]
        overlay = 0.55 * (img_np / 255.0) + 0.45 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"PneumoNet Grad-CAM | True: {true_label} | Pred: {pred_label} ({confidence * 100:.1f}%)",
            fontsize=13, fontweight="bold"
        )

        axes[0].imshow(img_np)
        axes[0].set_title("Original X-Ray", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Pathology Map", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("Pathology Overlay", fontsize=11)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Grad-CAM saved -> {save_path}")


# ==============================================================================
# Model Evaluation Pipeline
# ==============================================================================

def load_trained_model(checkpoint_path, device):
    """Loads PneumoNet from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
def load_trained_model(checkpoint_path, device):
    """Loads trained checkpoint state dict into PneumoNet."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = DualPathNet(
        num_classes=ckpt.get("num_classes", 2),
        model_name=ckpt.get("model_name", "PneumoNet")
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    best_threshold = ckpt.get("best_threshold", 0.5)
    print(f"Successfully loaded {model.model_name} from {checkpoint_path} (decision threshold: {best_threshold:.3f})")
    return model, ckpt.get("history", {}), best_threshold


def evaluate_test_set(model, test_loader, class_names, device, results_dir="results", use_tta=True, threshold=0.5):
    """
    Evaluates model across test dataset, logs all standard clinical metrics,
    applies TTA and calibrated decision threshold, and plots Confusion Matrix + ROC Curve.
    """
    os.makedirs(results_dir, exist_ok=True)
    model.eval().to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"\nEvaluating on Test Set (TTA: {use_tta}, Decision Threshold: {threshold:.3f})...")
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)

            if use_tta:
                outputs_orig = model(imgs)
                outputs_flip = model(torch.flip(imgs, dims=[3]))

                h, w = imgs.shape[2], imgs.shape[3]
                crop_margin = 6
                imgs_crop = F.interpolate(
                    imgs[:, :, crop_margin:h-crop_margin, crop_margin:w-crop_margin],
                    size=(h, w), mode="bilinear", align_corners=False
                )
                outputs_crop = model(imgs_crop)
                outputs_crop_flip = model(torch.flip(imgs_crop, dims=[3]))

                p_orig = F.softmax(outputs_orig, dim=1)[:, 1]
                p_flip = F.softmax(outputs_flip, dim=1)[:, 1]
                p_crop = F.softmax(outputs_crop, dim=1)[:, 1]
                p_crop_flip = F.softmax(outputs_crop_flip, dim=1)[:, 1]

                probs_pos = 0.40 * p_orig + 0.30 * p_flip + 0.15 * p_crop + 0.15 * p_crop_flip
            else:
                outputs = model(imgs)
                probs_pos = F.softmax(outputs, dim=1)[:, 1]

            preds = (probs_pos >= threshold).long()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs_pos.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute metrics
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average="binary")
    recall = recall_score(all_targets, all_preds, average="binary")
    f1 = f1_score(all_targets, all_preds, average="binary")
    auc = roc_auc_score(all_targets, all_probs)

    # Compute sensitivity & specificity
    cm = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 65)
    print("  PNEUMONET — CLINICAL EVALUATION REPORT (TEST SET)")
    print("=" * 65)
    print(f"  Total Test Samples   : {len(all_targets)}")
    print(f"  Decision Threshold   : {threshold:.3f}")
    print(f"  Overall Accuracy     : {acc * 100:.2f}%")
    print(f"  Sensitivity (Recall) : {sensitivity * 100:.2f}%  (Pneumonia Detection)")
    print(f"  Specificity          : {specificity * 100:.2f}%  (Normal Identification)")
    print(f"  Precision            : {precision * 100:.2f}%")
    print(f"  F1-Score             : {f1 * 100:.2f}%")
    print(f"  AUC-ROC              : {auc:.4f}")
    print("=" * 65)
    print("\nDetailed Classification Report:")
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4))

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"PneumoNet Confusion Matrix\nAcc: {acc*100:.2f}% | Sensitivity: {sensitivity*100:.2f}%", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "PneumoNet_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"PneumoNet (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("PneumoNet — ROC Curve", fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "PneumoNet_roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved -> {roc_path}")

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc
    }


def generate_sample_gradcams(model, test_dir, class_names, device, results_dir="results", samples_per_class=2):
    """
    Generates Grad-CAM visualizations on test images.
    """
    os.makedirs(results_dir, exist_ok=True)
    _, eval_tf = get_transforms()
    gradcam = GradCAM(model, device)

    print("\nGenerating Grad-CAM visualizations for test samples...")
    for class_name in class_names:
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        image_files = sorted(glob.glob(os.path.join(class_folder, "*.*")))
        selected_files = image_files[:samples_per_class]

        for i, img_path in enumerate(selected_files, 1):
            try:
                img_pil = Image.open(img_path).convert("RGB")
                img_tensor = eval_tf(img_pil).unsqueeze(0)  # (1, 3, 224, 224)

                heatmap, pred_idx, confidence = gradcam.generate(img_tensor)
                pred_label = class_names[pred_idx]

                save_filename = f"PneumoNet_gradcam_{class_name}_sample{i}.png"
                save_path = os.path.join(results_dir, save_filename)
                gradcam.save_visualization(
                    img_pil, heatmap,
                    pred_label=pred_label,
                    true_label=class_name,
                    confidence=confidence,
                    save_path=save_path
                )
            except Exception as e:
                print(f"  Warning: failed to generate Grad-CAM for {img_path}: {e}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate PneumoNet on Chest X-Ray dataset.")
    parser.add_argument("--data_dir", type=str, default="chest_xray", help="Path to chest_xray directory")
    parser.add_argument("--checkpoint", type=str, default=r"checkpoints\PneumoNet_best.pth")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--threads", type=int, default=10, help="PyTorch CPU threads")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    dataset_root = args.data_dir
    if not os.path.exists(dataset_root):
        dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chest_xray")

    test_dir = os.path.join(dataset_root, "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(dataset_root, "chest_xray", "test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device} (threads: {torch.get_num_threads()})")

    # Load Model
    model, history, best_thresh = load_trained_model(args.checkpoint, device)

    # Load Test Data
    _, eval_tf = get_transforms()
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    class_names = test_dataset.classes

    # Evaluate with calibrated threshold and TTA
    metrics = evaluate_test_set(
        model, test_loader, class_names, device,
        results_dir=args.results_dir,
        use_tta=True,
        threshold=best_thresh
    )

    # Generate Grad-CAM Samples
    generate_sample_gradcams(model, test_dir, class_names, device, results_dir=args.results_dir, samples_per_class=args.samples_per_class)

    print("\n[OK] Evaluation & Grad-CAM pipeline complete!")


if __name__ == "__main__":
    main()
