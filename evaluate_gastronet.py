"""
MediScan AI — GastroNet Evaluation & Grad-CAM Pipeline
Evaluates best checkpoint on HyperKvasir Endoscopy test split,
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

# Import model architecture and data utilities from training script
from train_gastronet import (
    GastroNet, get_transforms, collect_samples, CustomSampleDataset,
    CLASS_NAMES, ULCER_FOLDERS, NORMAL_FOLDERS
)
from sklearn.model_selection import train_test_split


# ==============================================================================
# Grad-CAM Engine
# ==============================================================================

class GradCAM:
    """
    Grad-CAM Heatmap Generator for GastroNet (DualPathResNet):
    Hooks into the final residual convolution block inside the refinement stage.
    """
    def __init__(self, model: GastroNet, device: torch.device):
        self.model = model
        self.device = device
        self.gradients = None
        self.activations = None
        self.model.eval()
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.refine[2]

        def fwd_hook(module, inp, out):
            self.activations = out.detach()

        def bwd_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        target_layer.register_forward_hook(fwd_hook)
        target_layer.register_full_backward_hook(bwd_hook)

    def generate(self, img_tensor, target_class=None):
        self.model.zero_grad()
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)

        outputs = self.model(img_tensor)
        probs = F.softmax(outputs, dim=1)

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        conf = probs[0, target_class].item()

        score = outputs[0, target_class]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, target_class, conf

    def save_visualization(self, img_pil, heatmap, pred_label, true_label,
                           confidence, save_path):
        img_resized = img_pil.resize((224, 224))
        img_np = np.array(img_resized)

        heatmap_color = cm.jet(heatmap)[:, :, :3]
        overlay = 0.55 * (img_np / 255.0) + 0.45 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"GastroNet Grad-CAM | True: {true_label} | Pred: {pred_label} ({confidence * 100:.1f}%)",
            fontsize=13, fontweight="bold"
        )

        axes[0].imshow(img_np)
        axes[0].set_title("Original Endoscopy", fontsize=11)
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
        print(f"  Grad-CAM saved -> {save_path}", flush=True)


# ==============================================================================
# Model Evaluation Pipeline
# ==============================================================================

def load_trained_model(checkpoint_path, device):
    """Loads GastroNet from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GastroNet(
        num_classes=ckpt.get("num_classes", 2),
        model_name=ckpt.get("model_name", "GastroNet")
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    best_threshold = ckpt.get("best_threshold", 0.5)
    print(f"Successfully loaded {model.model_name} from {checkpoint_path} (decision threshold: {best_threshold:.3f})", flush=True)
    return model, ckpt.get("history", {}), ckpt.get("class_names", CLASS_NAMES), best_threshold


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

    print(f"\nEvaluating on Test Set (TTA: {use_tta}, Decision Threshold: {threshold:.3f})...", flush=True)
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
    cm_mat = confusion_matrix(all_targets, all_preds)
    tn, fp, fn, tp = cm_mat.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "=" * 65, flush=True)
    print("  GASTRONET — CLINICAL EVALUATION REPORT (TEST SET)", flush=True)
    print("=" * 65, flush=True)
    print(f"  Total Test Samples   : {len(all_targets)}", flush=True)
    print(f"  Decision Threshold   : {threshold:.3f}", flush=True)
    print(f"  Overall Accuracy     : {acc * 100:.2f}%", flush=True)
    print(f"  Sensitivity (Recall) : {sensitivity * 100:.2f}%  (Ulcer Detection)", flush=True)
    print(f"  Specificity          : {specificity * 100:.2f}%  (Normal Identification)", flush=True)
    print(f"  Precision            : {precision * 100:.2f}%", flush=True)
    print(f"  F1-Score             : {f1 * 100:.2f}%", flush=True)
    print(f"  AUC-ROC              : {auc:.4f}", flush=True)
    print("=" * 65, flush=True)
    print("\nDetailed Classification Report:", flush=True)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4), flush=True)

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, annot_kws={"size": 14, "weight": "bold"}
    )
    plt.title(f"GastroNet Confusion Matrix\nAcc: {acc*100:.2f}% | Sensitivity: {sensitivity*100:.2f}%", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("True Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "GastroNet_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}", flush=True)

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"GastroNet (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    plt.title("GastroNet — ROC Curve", fontsize=12)
    plt.legend(loc="lower right", frameon=True)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "GastroNet_roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC curve saved -> {roc_path}", flush=True)

    return {
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc": auc
    }


def generate_sample_gradcams(model, dataset_root, class_names, device,
                             results_dir="results", samples_per_class=2, seed=42):
    """
    Generates Grad-CAM visualizations on test images from the HyperKvasir dataset.
    Uses the same test split as training to ensure we only visualize holdout images.
    """
    os.makedirs(results_dir, exist_ok=True)
    _, eval_tf = get_transforms()
    gradcam = GradCAM(model, device)

    # Reconstruct the test split to grab actual test images
    all_samples = collect_samples(dataset_root)
    labels = [s[1] for s in all_samples]
    _, test_samples = train_test_split(
        all_samples, test_size=0.15, stratify=labels, random_state=seed
    )

    print("\n" + "=" * 65, flush=True)
    print("  GENERATING GRAD-CAM EXPLAINABILITY VISUALIZATIONS", flush=True)
    print("=" * 65, flush=True)

    for class_idx, class_label in enumerate(class_names):
        class_test_samples = [s for s in test_samples if s[1] == class_idx]
        selected = class_test_samples[:samples_per_class]

        for i, (img_path, true_label_idx) in enumerate(selected, 1):
            try:
                img_pil = Image.open(img_path).convert("RGB")
                img_tensor = eval_tf(img_pil).unsqueeze(0)

                heatmap, pred_idx, confidence = gradcam.generate(img_tensor)
                pred_label = class_names[pred_idx]

                save_filename = f"GastroNet_gradcam_{class_label}_sample{i}.png"
                save_path = os.path.join(results_dir, save_filename)
                gradcam.save_visualization(
                    img_pil, heatmap,
                    pred_label=pred_label,
                    true_label=class_label,
                    confidence=confidence,
                    save_path=save_path
                )
            except Exception as e:
                print(f"  Warning: failed to generate Grad-CAM for {img_path}: {e}", flush=True)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GastroNet on HyperKvasir Endoscopy dataset.")
    parser.add_argument("--data_dir", type=str, default=r"labeled-images")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/GastroNet_best.pth")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--threads", type=int, default=10, help="PyTorch CPU threads")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (threads: {torch.get_num_threads()})", flush=True)

    model, history, saved_classes, best_thresh = load_trained_model(args.checkpoint, device)

    # Build test loader from the same split used during training
    _, eval_tf = get_transforms()
    all_samples = collect_samples(args.data_dir)
    labels = [s[1] for s in all_samples]
    _, test_samples = train_test_split(
        all_samples, test_size=0.15, stratify=labels, random_state=42
    )
    test_dataset = CustomSampleDataset(test_samples, transform=eval_tf)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    class_names = saved_classes if saved_classes else CLASS_NAMES

    # Evaluate with calibrated threshold and TTA
    metrics = evaluate_test_set(
        model, test_loader, class_names, device,
        results_dir=args.results_dir,
        use_tta=True,
        threshold=best_thresh
    )

    # Generate Grad-CAM Samples
    generate_sample_gradcams(
        model, args.data_dir, class_names, device,
        results_dir=args.results_dir,
        samples_per_class=args.samples_per_class
    )

    print("\n[OK] GastroNet evaluation and explainability pipeline complete!", flush=True)


if __name__ == "__main__":
    main()
