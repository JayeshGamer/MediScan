import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Configure publication-grade styling matching IEEE transactions/conferences
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.titlesize": 9,
    "lines.linewidth": 1.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

os.makedirs("paper_figures", exist_ok=True)

# ------------------------------------------------------------------------------
# 1. Multi-Panel Training Curves (Loss & Accuracy across all 3 models)
# ------------------------------------------------------------------------------
ckpt_t = torch.load("checkpoints/TumorNet_best.pth", map_location="cpu", weights_only=False)
ckpt_p = torch.load("checkpoints/PneumoNet_best.pth", map_location="cpu", weights_only=False)
ckpt_g = torch.load("checkpoints/GastroNet_best.pth", map_location="cpu", weights_only=False)

h_t = ckpt_t.get("history", {})
h_p = ckpt_p.get("history", {})
h_g = ckpt_g.get("history", {})

fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.0), dpi=300)

models_data = [
    ("TumorNet (Brain MRI)", h_t, ["val_acc", "val_f1"]),
    ("PneumoNet (Pediatric CXR)", h_p, ["val_acc", "val_auc"]),
    ("GastroNet (Endoscopy)", h_g, ["val_acc", "val_auc"])
]

for col, (title, h, extra_metrics) in enumerate(models_data):
    epochs = range(1, len(h["train_loss"]) + 1)
    
    # Top row: Loss
    ax_loss = axes[0, col]
    ax_loss.plot(epochs, h["train_loss"], label="Train Loss", color="#1f77b4")
    ax_loss.plot(epochs, h["val_loss"], label="Val Loss", color="#d62728", linestyle="--")
    ax_loss.set_title(title, fontweight="bold")
    ax_loss.set_ylabel("Loss" if col == 0 else "")
    ax_loss.legend(loc="upper right", frameon=True, framealpha=0.8)
    ax_loss.set_xlim(1, max(epochs))
    
    # Bottom row: Accuracy / Metrics
    ax_acc = axes[1, col]
    ax_acc.plot(epochs, [x*100 for x in h["train_acc"]], label="Train Acc", color="#2ca02c")
    ax_acc.plot(epochs, [x*100 for x in h["val_acc"]], label="Val Acc", color="#9467bd", linestyle="--")
    if extra_metrics[1] in h:
        metric_name = extra_metrics[1].replace("val_", "Val ").upper()
        ax_acc.plot(epochs, [x*100 for x in h[extra_metrics[1]]], label=metric_name, color="#ff7f0e", linestyle=":")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Metric (%)" if col == 0 else "")
    ax_acc.legend(loc="lower right", frameon=True, framealpha=0.8)
    ax_acc.set_xlim(1, max(epochs))
    ax_acc.set_ylim(40, 102)

plt.tight_layout()
plt.savefig("paper_figures/fig_training_curves_multipanel.png", dpi=300, bbox_inches="tight")
plt.savefig("paper_figures/fig_training_curves_multipanel.pdf", bbox_inches="tight")
plt.close()
print("Saved multi-panel training curves -> paper_figures/fig_training_curves_multipanel.png")


# ------------------------------------------------------------------------------
# 2. Multi-Panel Confusion Matrices
# ------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.6), dpi=300)

# (a) TumorNet
cm_t = np.array([
    [332, 42, 25, 1],
    [2, 390, 4, 4],
    [0, 0, 400, 0],
    [1, 2, 0, 397]
])
classes_t = ["Glioma", "Mening.", "No Tumor", "Pituit."]
sns.heatmap(cm_t, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes_t, yticklabels=classes_t, ax=axes[0],
            annot_kws={"size": 7.5})
axes[0].set_title("(a) TumorNet (Test N=1,600)", fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Ground Truth")

# (b) PneumoNet (tau=0.780)
cm_p = np.array([
    [215, 19],
    [46, 344]
])
classes_p = ["Normal", "Pneumonia"]
sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes_p, yticklabels=classes_p, ax=axes[1],
            annot_kws={"size": 8.5})
axes[1].set_title(r"(b) PneumoNet ($\tau=0.78$, N=624)", fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("")

# (c) GastroNet (tau=0.580)
cm_g = np.array([
    [460, 15],
    [13, 223]
])
classes_g = ["Normal", "Ulcer"]
sns.heatmap(cm_g, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=classes_g, yticklabels=classes_g, ax=axes[2],
            annot_kws={"size": 8.5})
axes[2].set_title(r"(c) GastroNet ($\tau=0.58$, N=711)", fontweight="bold")
axes[2].set_xlabel("Predicted")
axes[2].set_ylabel("")

plt.tight_layout()
plt.savefig("paper_figures/fig_confusion_matrices_multipanel.png", dpi=300, bbox_inches="tight")
plt.savefig("paper_figures/fig_confusion_matrices_multipanel.pdf", bbox_inches="tight")
plt.close()
print("Saved multi-panel confusion matrices -> paper_figures/fig_confusion_matrices_multipanel.png")


# ------------------------------------------------------------------------------
# 3. Multi-Panel ROC Curves
# ------------------------------------------------------------------------------
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from evaluate_tumornet import load_trained_model as load_t, get_transforms as get_tf_t
from evaluate_pneumonet import load_trained_model as load_p, get_transforms as get_tf_p
from evaluate_gastronet import load_trained_model as load_g, get_transforms as get_tf_g, collect_samples, CustomSampleDataset
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), dpi=300)

# (a) TumorNet ROC
model_t, _, _ = load_t("checkpoints/TumorNet_best.pth", device)
_, eval_tf_t = get_tf_t()
ds_t = datasets.ImageFolder("Brain Tumor MRI Dataset/Testing", transform=eval_tf_t)
loader_t = DataLoader(ds_t, batch_size=32, shuffle=False)
probs_t, targets_t = [], []
with torch.no_grad():
    for imgs, tgts in loader_t:
        probs = F.softmax(model_t(imgs.to(device)), dim=1)
        probs_t.extend(probs.cpu().numpy())
        targets_t.extend(tgts.numpy())
probs_t = np.array(probs_t)
targets_t = np.array(targets_t)
y_bin_t = label_binarize(targets_t, classes=[0, 1, 2, 3])
auc_macro_t = roc_auc_score(y_bin_t, probs_t, average="macro")

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
labels_t = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
for i in range(4):
    fpr, tpr, _ = roc_curve(y_bin_t[:, i], probs_t[:, i])
    auc_val = roc_auc_score(y_bin_t[:, i], probs_t[:, i])
    axes[0].plot(fpr, tpr, color=colors[i], label=f"{labels_t[i]} ({auc_val:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=1.0)
axes[0].set_title(f"(a) TumorNet (Macro AUC={auc_macro_t:.3f})", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right", frameon=True, framealpha=0.8, fontsize=6.5)

# (b) PneumoNet ROC
model_p, _, _ = load_p("checkpoints/PneumoNet_best.pth", device)
_, eval_tf_p = get_tf_p()
ds_p = datasets.ImageFolder("chest_xray/test", transform=eval_tf_p)
loader_p = DataLoader(ds_p, batch_size=32, shuffle=False)
probs_p, targets_p = [], []
with torch.no_grad():
    for imgs, tgts in loader_p:
        probs = F.softmax(model_p(imgs.to(device)), dim=1)[:, 1]
        probs_p.extend(probs.cpu().numpy())
        targets_p.extend(tgts.numpy())
probs_p = np.array(probs_p)
targets_p = np.array(targets_p)
fpr_p, tpr_p, _ = roc_curve(targets_p, probs_p)
auc_p = roc_auc_score(targets_p, probs_p)
axes[1].plot(fpr_p, tpr_p, color="#d62728", label=f"Pneumonia (AUC={auc_p:.4f})")
# Plot operating points
axes[1].plot(19/234, 344/390, marker='o', markersize=6, color='blue', label=r'$\tau=0.78$ (Operating Pt)')
axes[1].plot(54/234, 375/390, marker='s', markersize=6, color='green', label=r'$\tau=0.50$ (High Sens.)')
axes[1].plot([0, 1], [0, 1], "k--", lw=1.0)
axes[1].set_title(f"(b) PneumoNet (AUC={auc_p:.3f})", fontweight="bold")
axes[1].set_xlabel("False Positive Rate")
axes[1].legend(loc="lower right", frameon=True, framealpha=0.8, fontsize=6.5)

# (c) GastroNet ROC
model_g, _, _, _ = load_g("checkpoints/GastroNet_best.pth", device)
_, eval_tf_g = get_tf_g()
all_s = collect_samples("labeled-images")
lbls = [s[1] for s in all_s]
_, test_s = train_test_split(all_s, test_size=0.15, stratify=lbls, random_state=42)
ds_g = CustomSampleDataset(test_s, transform=eval_tf_g)
loader_g = DataLoader(ds_g, batch_size=32, shuffle=False)
probs_g, targets_g = [], []
with torch.no_grad():
    for imgs, tgts in loader_g:
        probs = F.softmax(model_g(imgs.to(device)), dim=1)[:, 1]
        probs_g.extend(probs.cpu().numpy())
        targets_g.extend(tgts.numpy())
probs_g = np.array(probs_g)
targets_g = np.array(targets_g)
fpr_g, tpr_g, _ = roc_curve(targets_g, probs_g)
auc_g = roc_auc_score(targets_g, probs_g)
axes[2].plot(fpr_g, tpr_g, color="#ff7f0e", label=f"Ulcer (AUC={auc_g:.4f})")
axes[2].plot(15/475, 223/236, marker='o', markersize=6, color='blue', label=r'$\tau=0.58$ (Operating Pt)')
axes[2].plot(25/475, 225/236, marker='s', markersize=6, color='green', label=r'$\tau=0.50$ (Default)')
axes[2].plot([0, 1], [0, 1], "k--", lw=1.0)
axes[2].set_title(f"(c) GastroNet (AUC={auc_g:.3f})", fontweight="bold")
axes[2].set_xlabel("False Positive Rate")
axes[2].legend(loc="lower right", frameon=True, framealpha=0.8, fontsize=6.5)

plt.tight_layout()
plt.savefig("paper_figures/fig_roc_curves_multipanel.png", dpi=300, bbox_inches="tight")
plt.savefig("paper_figures/fig_roc_curves_multipanel.pdf", bbox_inches="tight")
plt.close()
print("Saved multi-panel ROC curves -> paper_figures/fig_roc_curves_multipanel.png")
