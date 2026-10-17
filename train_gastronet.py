"""
MediScan AI — GastroNet v2 Training Pipeline
Dataset: HyperKvasir Labeled Images (Ulcer / Normal)
Architecture: Dual-Path Residual CNN (3x3 Fine + 5x5 Broad) + CBAM Attention + Dual Pooling
100% Custom Model Built From Scratch (No Pre-trained Backbones)
"""

import os
import copy
import time
import glob
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
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
    """Path A: 3x3 Residual Blocks — captures fine-grained mucosal textures and ulcer margins."""
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
    """Path B: 5x5 Residual Blocks — captures broad mucosal structure and lesion morphology."""
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


class GastroNet(nn.Module):
    """
    MediScan AI — GastroNet v2 (Dual-Path Residual CNN + CBAM)
    Detects Gastric Ulcer Disease from endoscopy images.
    100% Custom Model Built From Scratch
    """
    def __init__(self, num_classes=2, in_channels=3, model_name="GastroNet"):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes

        # Dual Paths
        self.path_a = PathA(in_ch=in_channels)
        self.path_b = PathB(in_ch=in_channels)

        joined_ch = self.path_a.out_channels + self.path_b.out_channels  # 512

        # 1x1 Conv Join
        self.join_conv = nn.Sequential(
            nn.Conv2d(joined_ch, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True)
        )

        # CBAM Attention
        self.attention = CBAM(512, reduction=16)

        # Refinement with Residual Blocks
        self.refine = nn.Sequential(
            ResidualConvBlock(512, 512, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            ResidualConvBlock(512, 512, kernel_size=3, padding=1)
        )

        # Dual Pooling: Global Avg + Global Max Pooling (512 * 2 = 1024 features)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Modern Decision Head
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

        # Dual Pooling (Avg + Max)
        pooled_avg = self.avg_pool(refined)
        pooled_max = self.max_pool(refined)
        pooled = torch.cat([pooled_avg, pooled_max], dim=1)

        return self.classifier(pooled)


# ==============================================================================
# HyperKvasir Dataset — Ulcer vs Normal Mapping
# ==============================================================================

# Ulcer (positive) class: mucosal ulcerations/erosions across GI tract
ULCER_FOLDERS = [
    # Upper GI mucosal breaks
    os.path.join("upper-gi-tract", "pathological-findings", "esophagitis-a"),
    os.path.join("upper-gi-tract", "pathological-findings", "esophagitis-b-d"),
    os.path.join("upper-gi-tract", "pathological-findings", "barretts"),
    os.path.join("upper-gi-tract", "pathological-findings", "barretts-short-segment"),
    # Lower GI ulcerative colitis
    os.path.join("lower-gi-tract", "pathological-findings", "ulcerative-colitis-grade-1"),
    os.path.join("lower-gi-tract", "pathological-findings", "ulcerative-colitis-grade-2"),
    os.path.join("lower-gi-tract", "pathological-findings", "ulcerative-colitis-grade-3"),
    os.path.join("lower-gi-tract", "pathological-findings", "ulcerative-colitis-grade-1-2"),
    os.path.join("lower-gi-tract", "pathological-findings", "ulcerative-colitis-grade-2-3"),
]

# Normal (negative) class: healthy mucosal landmarks across Upper and Lower GI
NORMAL_FOLDERS = [
    # Upper GI normal landmarks
    os.path.join("upper-gi-tract", "anatomical-landmarks", "pylorus"),
    os.path.join("upper-gi-tract", "anatomical-landmarks", "retroflex-stomach"),
    # Lower GI normal landmarks
    os.path.join("lower-gi-tract", "anatomical-landmarks", "cecum"),
    os.path.join("lower-gi-tract", "anatomical-landmarks", "retroflex-rectum"),
]

CLASS_NAMES = ["Normal", "Ulcer"]


def collect_samples(dataset_root):
    """
    Collect all (image_path, label) tuples from HyperKvasir subfolders.
    Label 0 = Normal, Label 1 = Ulcer
    """
    samples = []

    # Normal class (label = 0)
    for folder in NORMAL_FOLDERS:
        full_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(full_path):
            continue
        for img_file in sorted(glob.glob(os.path.join(full_path, "*.jpg"))):
            samples.append((img_file, 0))

    # Ulcer class (label = 1)
    for folder in ULCER_FOLDERS:
        full_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(full_path):
            continue
        for img_file in sorted(glob.glob(os.path.join(full_path, "*.jpg"))):
            samples.append((img_file, 1))

    return samples


# ==============================================================================
# Advanced Medical Loss & Regularization Functions
# ==============================================================================

class ClassWeightedFocalLoss(nn.Module):
    """
    Focal Loss with class-frequency balancing and label smoothing.
    Addresses endoscopy reflections, lighting shifts, and subtle mucosal breaks
    by focusing gradient updates on hard boundary examples.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.04, reduction="mean"):
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
    """Endoscopy-tailored data augmentation with illumination and lesion margin preservation."""
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06)),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
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


class CustomSampleDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = Image.open(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


def build_dataloaders(dataset_root, batch_size=32, samples_per_epoch=0, val_samples=0, seed=42):
    train_tf, eval_tf = get_transforms()

    all_samples = collect_samples(dataset_root)
    if len(all_samples) == 0:
        raise RuntimeError(f"No images found in {dataset_root}. Check ULCER_FOLDERS and NORMAL_FOLDERS paths.")

    labels = [s[1] for s in all_samples]

    # Stratified split: 70% train, 15% val, 15% test
    train_val_samples, test_samples = train_test_split(
        all_samples, test_size=0.15, stratify=labels, random_state=seed
    )
    train_val_labels = [s[1] for s in train_val_samples]
    # 15% of original = 0.1765 of remaining 85%
    train_samples, val_samples_list = train_test_split(
        train_val_samples, test_size=0.1765, stratify=train_val_labels, random_state=seed
    )

    train_dataset = CustomSampleDataset(train_samples, transform=train_tf)
    val_dataset = CustomSampleDataset(val_samples_list, transform=eval_tf)
    test_dataset = CustomSampleDataset(test_samples, transform=eval_tf)

    # Balanced sampler across Normal and Ulcer
    train_targets = [s[1] for s in train_samples]
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / class_counts.astype(np.float32)
    sample_weights = [class_weights[t] for t in train_targets]

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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    print("\n" + "=" * 65, flush=True)
    print(f"  Dataset Root         : {dataset_root}", flush=True)
    print(f"  Classes              : {CLASS_NAMES}", flush=True)
    print(f"  Total Images Found   : {len(all_samples):,}", flush=True)
    print(f"  Train Pool Size      : {len(train_dataset):,} images (Normal: {class_counts[0]}, Ulcer: {class_counts[1]})", flush=True)
    print(f"  Samples per Epoch    : {effective_samples} balanced images", flush=True)
    print(f"  Val Set Size         : {len(val_subset)} images", flush=True)
    print(f"  Test Set Size        : {len(test_dataset)} images", flush=True)
    print("=" * 65, flush=True)

    return train_loader, val_loader, test_loader, CLASS_NAMES


# ==============================================================================
# Early Stopping & Training
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

    def __call__(self, val_loss, val_acc, val_auc, model):
        composite_score = (val_auc if val_auc and not np.isnan(val_auc) else val_acc) * 0.5 + val_acc * 0.5
        improved = False
        if composite_score > self.best_score + self.delta:
            self.best_score = composite_score
            improved = True
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss

        if improved:
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "model_state": self.best_weights,
                "model_name": getattr(model, "model_name", "GastroNet"),
                "num_classes": getattr(model, "num_classes", 2),
                "class_names": CLASS_NAMES,
                "best_threshold": 0.5,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_loss": val_loss
            }, "checkpoints/GastroNet_best.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return improved


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return max(min_lr / base_lr, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_gastronet(model, train_loader, val_loader,
                    device, num_epochs=20, lr=4e-4, weight_decay=1e-4,
                    save_dir="checkpoints", resume_path=""):
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    # Focal Loss handles mucosal reflection and subtle ulcer edges with alpha weighting
    class_alpha = torch.tensor([1.0, 1.35], device=device)
    criterion = ClassWeightedFocalLoss(alpha=class_alpha, gamma=2.0, label_smoothing=0.03)
    eval_criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=2, total_epochs=num_epochs, base_lr=lr)
    early_stop = EarlyStopping(patience=8)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_start = max(1, int(num_epochs * 0.65))

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "val_auc": []
    }

    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    print(f"\n{'='*70}", flush=True)
    print(f"  Starting Enhanced Training: {model.model_name} on {device}", flush=True)
    print(f"  Epochs: {num_epochs} | Initial LR: {lr} | Weight Decay: {weight_decay} | AMP: {use_amp}", flush=True)
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

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        # ─── VALIDATION PHASE ───
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        all_val_probs, all_val_targets = [], []

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = model(imgs)
                    loss = eval_criterion(outputs, targets)

                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                v_loss += loss.item() * imgs.size(0)
                v_correct += (outputs.argmax(dim=1) == targets).sum().item()
                v_total += imgs.size(0)

                all_val_probs.extend(probs)
                all_val_targets.extend(targets.cpu().numpy())

        v_loss /= v_total
        v_acc = v_correct / v_total
        try:
            v_auc = roc_auc_score(all_val_targets, all_val_probs)
        except Exception:
            v_auc = float("nan")

        scheduler.step()
        epoch_dur = time.time() - epoch_start

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)
        history["val_auc"].append(v_auc)

        current_lr = scheduler.get_last_lr()[0]
        print(f"  >>> [Summary Epoch {epoch:02d}] Train Loss: {t_loss:.4f}, Acc: {t_acc*100:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Acc: {v_acc*100:.2f}%, AUC: {v_auc:.4f} | "
              f"Time: {epoch_dur:.1f}s | LR: {current_lr:.6f}\n", flush=True)

        early_stop(v_loss, v_acc, v_auc, model)
        if early_stop.stop:
            print(f"  [Early Stopping] Triggered at epoch {epoch}.", flush=True)
            break

    total_time = time.time() - start_total_time
    print(f"{'='*70}", flush=True)
    print(f"  Training finished in {total_time/60:.2f} minutes.", flush=True)
    print(f"{'='*70}", flush=True)

    best_weights = early_stop.best_weights if early_stop.best_weights is not None else model.state_dict()
    model.load_state_dict(best_weights)

    # Threshold calibration on validation set
    best_threshold = 0.5
    best_j = -1.0
    all_val_targets_np = np.array(all_val_targets)
    all_val_probs_np = np.array(all_val_probs)
    for thresh in np.linspace(0.35, 0.65, 31):
        preds_t = (all_val_probs_np >= thresh).astype(int)
        cm = confusion_matrix(all_val_targets_np, preds_t)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            j_stat = sens + spec - 1.0
            if j_stat > best_j:
                best_j = j_stat
                best_threshold = float(thresh)
    print(f"  Optimal validation decision threshold calibrated: {best_threshold:.3f} (Youden's J: {best_j:.4f})", flush=True)

    ckpt_path = os.path.join(save_dir, f"{model.model_name}_best.pth")
    torch.save({
        "model_state": model.state_dict(),
        "model_name": model.model_name,
        "num_classes": model.num_classes,
        "history": history,
        "class_names": CLASS_NAMES,
        "best_threshold": best_threshold
    }, ckpt_path)
    print(f"  Saved best model checkpoint -> {ckpt_path}\n", flush=True)

    return model, history


def plot_curves(history, model_name="GastroNet", save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name} (DualPathResNet) — Enhanced Training Progress", fontsize=14, fontweight="bold")

    # Loss Curve
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r-o", markersize=4, label="Val Loss")
    ax1.set_title("Focal / Cross-Entropy Loss", fontsize=12)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)

    # Accuracy Curve
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "b-o", markersize=4, label="Train Acc (%)")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "r-o", markersize=4, label="Val Acc (%)")
    if "val_auc" in history and any(not np.isnan(x) for x in history["val_auc"]):
        ax2.plot(epochs, [a * 100 for a in history["val_auc"]], "g-s", markersize=4, label="Val AUC (%)")
    ax2.set_title("Accuracy & AUC (%)", fontsize=12)
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
    parser = argparse.ArgumentParser(description="Train GastroNet on HyperKvasir Endoscopy dataset.")
    parser.add_argument("--data_dir", type=str,
                        default=r"labeled-images",
                        help="Path to HyperKvasir labeled-images dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Maximum epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--samples_per_epoch", type=int, default=0, help="Balanced samples per epoch (0 = full pool)")
    parser.add_argument("--val_samples", type=int, default=0, help="Validation samples (0 = full val set)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--threads", type=int, default=10, help="PyTorch CPU threads")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory for checkpoints")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for plots")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume training from")
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

    model = GastroNet(num_classes=len(class_names))
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Initialized {model.model_name} (DualPathResNet) with {total_params:,} total parameters.", flush=True)

    model, history = train_gastronet(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        resume_path=args.resume
    )

    plot_curves(history, model_name=model.model_name, save_dir=args.results_dir)
    print("\n[OK] GastroNet training complete!", flush=True)


if __name__ == "__main__":
    main()
