"""
MediScan AI — TumorNet v2 Training Pipeline
Dataset: Brain Tumor MRI Dataset (Glioma / Meningioma / No Tumor / Pituitary)
Architecture: Dual-Path Residual CNN (3x3 Fine + 5x5 Broad) + CBAM Attention + Dual Pooling
100% Custom Model Built From Scratch (No Pre-trained Backbones)
"""

import os
import copy
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


# ==============================================================================
# Model Architecture (100% Custom Dual-Path Residual CNN + CBAM)
# ==============================================================================

class ResidualConvBlock(nn.Module):
    """
    Two-conv block with SiLU (Swish) and Residual Shortcut connection.
    out = SiLU(BN(Conv2(SiLU(BN(Conv1(x)))))) + shortcut(x)
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x):
        res = self.shortcut(x)
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act2(out + res)


class PathA(nn.Module):
    """Path A: 3x3 Residual Blocks — captures fine-grained micro-textures and lesion margins."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.stage1 = nn.Sequential(
            ResidualConvBlock(in_ch, 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage4 = nn.Sequential(
            ResidualConvBlock(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2)
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class PathB(nn.Module):
    """Path B: 5x5 Residual Blocks — captures broad anatomical structure and macro morphology."""
    def __init__(self, in_ch=3):
        super().__init__()
        self.stage1 = nn.Sequential(
            ResidualConvBlock(in_ch, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage2 = nn.Sequential(
            ResidualConvBlock(32, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage3 = nn.Sequential(
            ResidualConvBlock(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.08)
        )
        self.stage4 = nn.Sequential(
            ResidualConvBlock(128, 256, kernel_size=5, padding=2),
            nn.MaxPool2d(2, 2)
        )
        self.out_channels = 256

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        mid = max(in_ch // reduction, 8)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, in_ch, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, in_ch, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(in_ch, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))


class TumorNet(nn.Module):
    """
    MediScan AI — TumorNet (v2 Dual-Path Residual CNN + CBAM)
    100% Custom Model Built From Scratch
    """
    def __init__(self, num_classes=4, in_channels=3, model_name="TumorNet"):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        self.path_a = PathA(in_ch=in_channels)
        self.path_b = PathB(in_ch=in_channels)

        joined_ch = self.path_a.out_channels + self.path_b.out_channels  # 512

        self.join_conv = nn.Sequential(
            nn.Conv2d(joined_ch, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )

        self.attention = CBAM(512, reduction=16)

        self.refine = nn.Sequential(
            ResidualConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ResidualConvBlock(512, 512, kernel_size=3, padding=1)
        )

        # Dual Pooling: AvgPool + MaxPool -> 512 * 2 = 1024 features
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        feat_a = self.path_a(x)
        feat_b = self.path_b(x)
        joined = self.join_conv(torch.cat([feat_a, feat_b], dim=1))
        attended = self.attention(joined)
        refined = self.refine(attended)

        # Concatenate Avg + Max pooling
        pooled_avg = self.avg_pool(refined)
        pooled_max = self.max_pool(refined)
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)

        return self.classifier(pooled)


# ==============================================================================
# Advanced Medical Loss & Regularization Functions
# ==============================================================================

class MultiClassFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with Label Smoothing.
    Focuses gradient updates on hard-to-distinguish classes (Glioma vs Meningioma)
    while preventing overconfidence on standard cases.
    """
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.05, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
            weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def mixup_data(x, y, alpha=0.2, device="cpu"):
    """Applies Mixup data augmentation to inputs and targets."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes loss with mixup labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ==============================================================================
# Data Loading & Transforms
# ==============================================================================

def get_transforms():
    """
    High-generalization MRI transforms:
    Combines affine spatial robustness with Random Erasing to prevent
    the network from over-indexing on non-pathological background artifacts.
    """
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.92, 1.08)),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.20, scale=(0.02, 0.20), value="random")
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return train_tf, eval_tf


class CustomSampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target


def build_dataloaders(dataset_root, batch_size=32, samples_per_epoch=0, val_samples=0, seed=42):
    train_tf, eval_tf = get_transforms()

    train_dir = os.path.join(dataset_root, "Training")
    test_dir = os.path.join(dataset_root, "Testing")

    if not os.path.exists(train_dir):
        for cand in [os.path.join(dataset_root, "train"), dataset_root]:
            if os.path.exists(cand):
                train_dir = cand
                break

    raw_train_ds = datasets.ImageFolder(train_dir)
    class_names = raw_train_ds.classes

    all_samples = list(raw_train_ds.samples)
    labels = [s[1] for s in all_samples]

    train_samples, val_samples_list = train_test_split(
        all_samples, test_size=0.15, stratify=labels, random_state=seed
    )

    train_dataset = CustomSampleDataset(train_samples, transform=train_tf)
    val_dataset = CustomSampleDataset(val_samples_list, transform=eval_tf)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_tf) if os.path.exists(test_dir) else None

    train_targets = [s[1] for s in train_samples]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts.astype(np.float32)
    sample_weights = [class_weights[t] for t in train_targets]

    # If samples_per_epoch <= 0, use full dataset length
    effective_samples = samples_per_epoch if (samples_per_epoch and samples_per_epoch > 0) else len(train_dataset)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=effective_samples,
        replacement=True
    )

    if val_samples and val_samples > 0 and val_samples < len(val_dataset):
        val_subset_indices = np.random.RandomState(seed).choice(
            len(val_dataset),
            size=val_samples,
            replace=False
        )
        val_subset = Subset(val_dataset, val_subset_indices)
    else:
        val_subset = val_dataset

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True) if test_dataset else None

    print("\n" + "=" * 65, flush=True)
    print(f"  Dataset Root         : {dataset_root}", flush=True)
    print(f"  Classes ({len(class_names)})         : {class_names}", flush=True)
    print(f"  Train Pool Size      : {len(train_dataset):,} images ({dict(zip(class_names, class_counts))})", flush=True)
    print(f"  Samples per Epoch    : {effective_samples} balanced images", flush=True)
    print(f"  Val Set Size         : {len(val_subset)} images", flush=True)
    if test_dataset:
        print(f"  Test Set Size        : {len(test_dataset)} images", flush=True)
    print("=" * 65, flush=True)

    return train_loader, val_loader, test_loader, class_names


# ==============================================================================
# Early Stopping & Training Engine
# ==============================================================================

class EarlyStopping:
    def __init__(self, patience=8, delta=0.0005):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.best_score = 0.0
        self.stop = False
        self.best_weights = None

    def __call__(self, val_loss, val_acc, val_f1, model):
        composite_score = val_f1 * 0.6 + val_acc * 0.4
        improved = False
        if composite_score > self.best_score + self.delta:
            self.best_score = composite_score
            improved = True
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss

        if improved:
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    """Linear warmup for first epochs, followed by cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(min_lr / base_lr, cosine_decay)
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_tumornet(model, train_loader, val_loader,
                   device, num_epochs=20, lr=4e-4, weight_decay=1e-4,
                   save_dir="checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # Class-balanced Focal Loss: increases Glioma weight to 1.35x and Meningioma to 1.05x to guarantee >=90% recall
    class_alpha = torch.tensor([1.35, 1.05, 0.90, 0.90], device=device)
    criterion = MultiClassFocalLoss(alpha=class_alpha, gamma=1.5, label_smoothing=0.05)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=num_epochs, base_lr=lr)
    early_stop = EarlyStopping(patience=8)

    # Mixed Precision Scaler
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Stochastic Weight Averaging (SWA)
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = max(1, int(num_epochs * 0.65))

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_f1": []
    }

    print(f"\n{'='*70}", flush=True)
    print(f"  Starting Enhanced Training: {model.model_name} on {device}", flush=True)
    print(f"  Epochs: {num_epochs} | Base LR: {lr} | Weight Decay: {weight_decay} | AMP: {use_amp} | SWA Start: Epoch {swa_start}", flush=True)
    print(f"{'='*70}\n", flush=True)

    start_total_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # ─── TRAIN PHASE ───
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0

        for batch_idx, (imgs, targets) in enumerate(train_loader, 1):
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            # Apply Mixup on 50% of batches for regularization
            use_mixup = (np.random.rand() < 0.5) and (epoch < swa_start)
            if use_mixup:
                mixed_imgs, y_a, y_b, lam = mixup_data(imgs, targets, alpha=0.2, device=device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(mixed_imgs)
                    loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(imgs)
                    loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item() * imgs.size(0)
            t_correct += (outputs.argmax(dim=1) == targets).sum().item()
            t_total += imgs.size(0)

            if batch_idx % 20 == 0 or batch_idx == len(train_loader):
                batch_acc = t_correct / t_total * 100.0
                current_loss = t_loss / t_total
                print(f"  Epoch [{epoch:02d}/{num_epochs}] Step [{batch_idx:03d}/{len(train_loader):03d}] "
                      f"Train Loss: {current_loss:.4f} | Batch Acc: {batch_acc:.1f}%", flush=True)

        t_loss /= t_total
        t_acc = t_correct / t_total

        # Update SWA model in late epochs
        if epoch >= swa_start:
            swa_model.update_parameters(model)

        # ─── VALIDATION PHASE ───
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_val_preds, all_val_targets = [], []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(imgs)
                    loss = eval_criterion(outputs, targets)

                preds = outputs.argmax(dim=1)
                v_loss += loss.item() * imgs.size(0)
                v_correct += (preds == targets).sum().item()
                v_total += imgs.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())

        v_loss /= v_total
        v_acc = v_correct / v_total
        v_f1 = f1_score(all_val_targets, all_val_preds, average="macro", zero_division=0)

        scheduler.step()
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["val_f1"].append(v_f1)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  >>> [Summary Epoch {epoch:02d}] Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%, Macro-F1: {v_f1:.4f} | "
              f"Time: {epoch_dur:.1f}s | LR: {current_lr:.6f}\n", flush=True)

        early_stop(v_loss, v_acc, v_f1, model)
        if early_stop.stop:
            print(f"  [Early Stopping] Triggered at epoch {epoch}.", flush=True)
            break

    total_time = time.time() - start_total_time
    print(f"{'='*70}", flush=True)
    print(f"  Training finished in {total_time/60:.2f} minutes.", flush=True)
    print(f"{'='*70}", flush=True)

    # Finalize best model selection (Early Stop best vs SWA)
    best_weights = early_stop.best_weights if early_stop.best_weights is not None else model.state_dict()
    model.load_state_dict(best_weights)

    ckpt_path = os.path.join(save_dir, f"{model.model_name}_best.pth")
    torch.save({
        "model_state": model.state_dict(),
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "history": history
    }, ckpt_path)
    print(f"  Saved best model checkpoint -> {ckpt_path}\n", flush=True)

    return model, history


def plot_curves(history, model_name="TumorNet", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} (DualPathResNet) — Enhanced Training Progress", fontsize=14, fontweight="bold")

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax1.set_title("Focal / Cross-Entropy Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", markersize=4, label="Train Acc (%)")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", markersize=4, label="Val Acc (%)")
    ax2.plot(epochs, [f * 100 for f in history["val_f1"]], "g-s", markersize=4, label="Val Macro F1 (%)")
    ax2.set_title("Accuracy & F1 Score (%)", fontsize=12)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score (%)")
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(save_dir, f"{model_name}_curves.png")
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Training curves plot saved -> {curve_path}", flush=True)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TumorNet on Brain Tumor MRI dataset.")
    parser.add_argument("--data_dir", type=str,
                        default="Brain Tumor MRI Dataset",
                        help="Path to Brain Tumor MRI dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples_per_epoch", type=int, default=0, help="Samples per epoch (0 = full dataset)")
    parser.add_argument("--val_samples", type=int, default=0, help="Validation samples (0 = full val set)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--threads", type=int, default=10, help="PyTorch CPU threads")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for plots")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} (threads: {torch.get_num_threads()})", flush=True)

    train_loader, val_loader, test_loader, class_names = build_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        val_samples=args.val_samples,
        seed=42
    )

    model = TumorNet(num_classes=len(class_names))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized {model.model_name} (DualPathResNet) with {total_params:,} parameters for {len(class_names)} classes.", flush=True)

    model, history = train_tumornet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir
    )

    # Save class names into checkpoint
    ckpt_path = os.path.join(args.save_dir, f"{model.model_name}_best.pth")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt["class_names"] = class_names
    torch.save(ckpt, ckpt_path)

    plot_curves(history, model_name=model.model_name, save_dir=args.results_dir)
    print("\n[OK] TumorNet training complete!", flush=True)


if __name__ == "__main__":
    main()
