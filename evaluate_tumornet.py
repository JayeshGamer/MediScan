"""
MediScan AI — TumorNet Evaluation & Grad-CAM Pipeline
Evaluates best checkpoint on Brain Tumor MRI test split,
supports Test-Time Augmentation (TTA), clinical classification metrics, and Grad-CAM.
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
from sklearn.preprocessing import label_binarize

from train_tumornet import TumorNet, get_transforms


# ==============================================================================
# Grad-CAM Engine
# ==============================================================================

class GradCAM:
    """
    Grad-CAM Heatmap Generator for TumorNet (DualPathResNet):
    Hooks into the final residual convolution block inside the refinement stage.
    """
    def __init__(self, model: TumorNet, device: torch.device):
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
            f"TumorNet Grad-CAM | True: {true_label} | Pred: {pred_label} ({confidence * 100:.1f}%)",
            fontsize=13, fontweight="bold"
        )

        axes[0].imshow(img_np)
        axes[0].set_title("Original MRI", fontsize=11)
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
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model = TumorNet(
        num_classes=ckpt.get("num_classes", 4),
        model_name=ckpt.get("model_name", "TumorNet")
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval().to(device)
    print(f"Successfully loaded {model.model_name} from {checkpoint_path}", flush=True)
    return model, ckpt.get("history", {}), ckpt.get("class_names", None)


def evaluate_test_set(model, test_loader, class_names, device, results_dir="results", use_tta=True):
    os.makedirs(results_dir, exist_ok=True)
    model.eval().to(device)

    all_preds = []
    all_targets = []
    all_probs = []

    print(f"\nRunning test set evaluation across {len(test_loader.dataset)} scans (TTA: {use_tta})...", flush=True)

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs = imgs.to(device)

            if use_tta:
                # Multi-Pass Test-Time Augmentation (Original, H-Flip, Zoom-Center, H-Flip-Zoom)
                outputs_orig = model(imgs)
                outputs_flip = model(torch.flip(imgs, dims=[3]))
                
                # Subtle zoom & crop
                h, w = imgs.shape[2], imgs.shape[3]
                crop_margin = 6
                imgs_crop = F.interpolate(
                    imgs[:, :, crop_margin:h-crop_margin, crop_margin:w-crop_margin],
                    size=(h, w), mode="bilinear", align_corners=False
                )
                outputs_crop = model(imgs_crop)
                outputs_crop_flip = model(torch.flip(imgs_crop, dims=[3]))

                probs_orig = F.softmax(outputs_orig, dim=1)
                probs_flip = F.softmax(outputs_flip, dim=1)
                probs_crop = F.softmax(outputs_crop, dim=1)
                probs_crop_flip = F.softmax(outputs_crop_flip, dim=1)

                probs = 0.40 * probs_orig + 0.30 * probs_flip + 0.15 * probs_crop + 0.15 * probs_crop_flip
            else:
                outputs = model(imgs)
                probs = F.softmax(outputs, dim=1)

            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_targets, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    try:
        y_bin = label_binarize(all_targets, classes=range(len(class_names)))
        auc = roc_auc_score(y_bin, all_probs, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    print("\n" + "=" * 65, flush=True)
    print("  CLINICAL TEST SET EVALUATION REPORT (TumorNet v2)", flush=True)
    print("=" * 65, flush=True)
    print(f"  Test Accuracy        : {acc * 100:.2f}%", flush=True)
    print(f"  Macro Precision      : {prec * 100:.2f}%", flush=True)
    print(f"  Macro Recall (Sens.) : {rec * 100:.2f}%", flush=True)
    print(f"  Macro F1-Score       : {f1 * 100:.2f}%", flush=True)
    print(f"  Macro ROC-AUC (OvR)  : {auc:.4f}", flush=True)
    print("=" * 65, flush=True)
    print("\nDetailed Classification Report:", flush=True)
    print(classification_report(all_targets, all_preds, target_names=class_names, digits=4), flush=True)

    # Confusion Matrix
    cm_mat = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_mat, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        cbar=False, annot_kws={"size": 13}
    )
    plt.title("TumorNet (DualPathResNet) — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.ylabel("Ground Truth", fontsize=11)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "TumorNet_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion Matrix saved -> {cm_path}", flush=True)

    # Multiclass ROC Curve (OvR)
    plt.figure(figsize=(7, 6))
    y_bin = label_binarize(all_targets, classes=range(len(class_names)))
    for i, cname in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
            class_auc = roc_auc_score(y_bin[:, i], all_probs[:, i])
            plt.plot(fpr, tpr, lw=2, label=f"{cname} (AUC = {class_auc:.3f})")
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess")
    plt.title(f"TumorNet — ROC Curves (Macro AUC = {auc:.3f})", fontsize=13, fontweight="bold")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "TumorNet_roc_curve.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ROC Curve saved -> {roc_path}", flush=True)

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


def run_gradcam_samples(model, dataset_root, class_names, device,
                        results_dir="results", samples_per_class=2):
    _, eval_tf = get_transforms()
    gradcam = GradCAM(model, device)

    test_dir = os.path.join(dataset_root, "Testing")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(dataset_root, "Training")

    print("\n" + "=" * 65, flush=True)
    print("  GENERATING GRAD-CAM EXPLAINABILITY VISUALIZATIONS", flush=True)
    print("=" * 65, flush=True)

    for class_name in class_names:
        class_folder = os.path.join(test_dir, class_name)
        if not os.path.exists(class_folder):
            continue

        image_files = sorted(glob.glob(os.path.join(class_folder, "*.*")))
        selected_files = image_files[:samples_per_class]

        for i, img_path in enumerate(selected_files, 1):
            try:
                img_pil = Image.open(img_path).convert("RGB")
                img_tensor = eval_tf(img_pil).unsqueeze(0)

                heatmap, pred_idx, confidence = gradcam.generate(img_tensor)
                pred_label = class_names[pred_idx]

                save_filename = f"TumorNet_gradcam_{class_name}_sample{i}.png"
                save_path = os.path.join(results_dir, save_filename)
                gradcam.save_visualization(
                    img_pil, heatmap,
                    pred_label=pred_label,
                    true_label=class_name,
                    confidence=confidence,
                    save_path=save_path
                )
            except Exception as e:
                print(f"  Warning: failed to generate Grad-CAM for {img_path}: {e}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate TumorNet v2 on Brain Tumor MRI dataset.")
    parser.add_argument("--data_dir", type=str,
                        default="Brain Tumor MRI Dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/TumorNet_best.pth")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_class", type=int, default=2)
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--threads", type=int, default=10, help="PyTorch CPU threads")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (threads: {torch.get_num_threads()})", flush=True)

    model, history, saved_classes = load_trained_model(args.checkpoint, device)

    test_dir = os.path.join(args.data_dir, "Testing")
    _, eval_tf = get_transforms()
    test_ds = datasets.ImageFolder(test_dir, transform=eval_tf)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    class_names = saved_classes if saved_classes else test_ds.classes

    evaluate_test_set(model, test_loader, class_names, device, results_dir=args.results_dir, use_tta=not args.no_tta)
    run_gradcam_samples(model, args.data_dir, class_names, device,
                        results_dir=args.results_dir,
                        samples_per_class=args.samples_per_class)
    print("\n[OK] TumorNet evaluation and explainability pipeline complete!", flush=True)


if __name__ == "__main__":
    main()
