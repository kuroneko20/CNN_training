"""
train.py
────────
Script huấn luyện EfficientNet-B0 với chiến lược fine-tuning 2 giai đoạn.
Đọc cấu hình từ configs/train.yaml, tự phát hiện dataset, train và lưu kết quả.

Cách dùng (trên Google Colab):
    python scripts/train.py --config configs/train.yaml
"""

import argparse
import copy
import os
import shutil
import time
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

try:
    from torchinfo import summary
    HAS_TORCHINFO = True
except ImportError:
    HAS_TORCHINFO = False


# ─── Dataset helpers ──────────────────────────────────────────────────

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def safe_pil_loader(path: str) -> Image.Image:
    """Load ảnh và convert sang RGB, xử lý đúng palette PNG có transparency."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            img = Image.open(f)
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            return img.convert("RGB")


def find_imagefolder_root(base: Path):
    def has_images(folder: Path) -> int:
        return sum(1 for f in folder.iterdir()
                   if f.is_file() and f.suffix.lower() in IMG_EXTS)

    def check_dir(d: Path):
        subdirs = [c for c in d.iterdir()
                   if c.is_dir()
                   and not c.name.startswith("__")
                   and not c.name.startswith(".")]
        if not subdirs:
            return None
        counts = {c.name: has_images(c) for c in subdirs}
        if sum(counts.values()) > 0:
            return d, {k: v for k, v in counts.items() if v > 0}
        for sub in subdirs:
            result = check_dir(sub)
            if result:
                return result
        return None

    return check_dir(base)


def find_splits(root: Path):
    split_names = ["train", "Train", "training", "val", "Val",
                   "validation", "test", "Test", "testing"]
    found = {s.lower(): root / s for s in split_names if (root / s).exists()}
    train = found.get("train") or found.get("training")
    val = (found.get("val") or found.get("validation")
           or found.get("test") or found.get("testing"))
    return train, val


# ─── Model builder ────────────────────────────────────────────────────

def build_model(cfg: dict, num_classes: int, device: str):
    """Tải EfficientNet-B0 pretrained và thay classifier head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)

    # Giai đoạn 1: đóng băng backbone
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features  # 1280
    hidden      = cfg["model"]["classifier_hidden"]
    d1          = cfg["model"]["dropout1"]
    d2          = cfg["model"]["dropout2"]

    model.classifier = nn.Sequential(
        nn.Dropout(p=d1, inplace=True),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=d2),
        nn.Linear(hidden, num_classes),
    )
    return model.to(device)


# ─── Training functions ───────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / total, 100.0 * correct / total


# ─── Plot helpers ─────────────────────────────────────────────────────

def plot_training_curves(history, unfreeze_epoch, num_epochs, avg_epoch, best_val_acc, save_path):
    ep  = range(1, num_epochs + 1)
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, history["train_loss"], "o-", color="#2196F3", lw=2, ms=4, label="Train Loss")
    ax.plot(ep, history["val_loss"],   "s--", color="#F44336", lw=2, ms=4, label="Val Loss")
    ax.axvline(unfreeze_epoch, color="gray", ls=":", alpha=0.7, label=f"Unfreeze @{unfreeze_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Loss Curve", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep, history["train_acc"], "o-", color="#4CAF50", lw=2, ms=4, label="Train Acc")
    ax.plot(ep, history["val_acc"],   "s--", color="#FF9800", lw=2, ms=4, label="Val Acc")
    ax.axhline(best_val_acc, color="#9C27B0", ls="--", alpha=0.8, label=f"Best {best_val_acc:.1f}%")
    ax.axvline(unfreeze_epoch, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Curve", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(ep, history["lr"], "o-", color="#795548", lw=2, ms=4)
    ax.axvline(unfreeze_epoch, color="gray", ls=":", alpha=0.7, label=f"Unfreeze @{unfreeze_epoch}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate (log)")
    ax.set_title("Learning Rate Schedule", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(ep, history["epoch_time"], color="#607D8B", alpha=0.7)
    ax.axhline(avg_epoch, color="red", ls="--", lw=2, label=f"Avg: {avg_epoch:.1f}s")
    ax.axvline(unfreeze_epoch, color="gray", ls=":", alpha=0.7)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Giây")
    ax.set_title("Thời Gian / Epoch", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)

    plt.suptitle("KẾT QUẢ HUẤN LUYỆN — EfficientNet-B0 (ImageNet pre-trained)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Đã lưu training_curves.png")


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm  = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (Validation Set)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Đã lưu confusion_matrix.png")


# ─── Main ─────────────────────────────────────────────────────────────

def main(args):
    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE.upper()}")

    # ── Dataset ──────────────────────────────────────────────────────
    DATA_ROOT = Path(cfg["dataset"]["data_root"])
    result = find_imagefolder_root(DATA_ROOT)
    if result is None:
        raise RuntimeError("❌ Không tìm thấy cấu trúc ImageFolder hợp lệ.")

    FOUND_ROOT, class_stats = result
    CLASS_NAMES = list(class_stats.keys())
    NUM_CLASSES = len(CLASS_NAMES)
    print(f"✅ Dataset: {NUM_CLASSES} lớp — {CLASS_NAMES}")

    # Tìm split
    train_split, val_split = find_splits(FOUND_ROOT.parent)
    if train_split is None:
        train_split, val_split = find_splits(FOUND_ROOT)
    HAS_SPLIT = train_split and val_split and train_split != val_split
    TRAIN_DIR  = str(train_split) if HAS_SPLIT else str(FOUND_ROOT)
    VAL_DIR    = str(val_split)   if HAS_SPLIT else str(FOUND_ROOT)

    # ── Transforms ───────────────────────────────────────────────────
    img_size = cfg["preprocessing"]["img_size"]
    mean     = cfg["preprocessing"]["imagenet_mean"]
    std      = cfg["preprocessing"]["imagenet_std"]
    aug      = cfg["augmentation"]

    train_tf = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=aug["horizontal_flip_prob"]),
        transforms.ColorJitter(
            brightness=aug["color_jitter"]["brightness"],
            contrast=aug["color_jitter"]["contrast"],
            saturation=aug["color_jitter"]["saturation"],
        ),
        transforms.RandomRotation(aug["rotation_degrees"]),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # ── Build DataLoader ─────────────────────────────────────────────
    dl_cfg  = cfg["dataloader"]
    BATCH   = dl_cfg["batch_size"]
    WORKERS = dl_cfg["num_workers"]
    SEED    = cfg["dataset"]["seed"]

    if HAS_SPLIT:
        train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf, loader=safe_pil_loader)
        val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_tf,   loader=safe_pil_loader)
    else:
        full_ds = datasets.ImageFolder(TRAIN_DIR, loader=safe_pil_loader)
        targets = np.array(full_ds.targets)
        val_ratio = cfg["dataset"]["val_split"]
        train_idx, val_idx = [], []
        rng = np.random.default_rng(SEED)
        for cls_id in np.unique(targets):
            idx = np.where(targets == cls_id)[0]
            rng.shuffle(idx)
            split = int(len(idx) * (1 - val_ratio))
            train_idx.extend(idx[:split].tolist())
            val_idx.extend(idx[split:].tolist())
        train_ds_base = datasets.ImageFolder(TRAIN_DIR, transform=train_tf, loader=safe_pil_loader)
        val_ds_base   = datasets.ImageFolder(TRAIN_DIR, transform=val_tf,   loader=safe_pil_loader)
        train_ds = Subset(train_ds_base, train_idx)
        val_ds   = Subset(val_ds_base,   val_idx)

    CLASS_NAMES = (train_ds.classes if hasattr(train_ds, "classes")
                   else train_ds.dataset.classes)
    NUM_CLASSES = len(CLASS_NAMES)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=WORKERS, pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=dl_cfg.get("persistent_workers", True),
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=WORKERS, pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=dl_cfg.get("persistent_workers", True),
        prefetch_factor=dl_cfg.get("prefetch_factor", 2),
    )
    print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Batch: {BATCH}")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(cfg, NUM_CLASSES, DEVICE)
    if HAS_TORCHINFO:
        summary(model, input_size=(1, 3, img_size, img_size), device=DEVICE,
                col_names=["input_size", "output_size", "num_params", "trainable"], depth=2)

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    tr_cfg         = cfg["training"]
    opt_cfg        = cfg["optimizer"]
    sched_cfg      = cfg["scheduler"]
    loss_cfg       = cfg["loss"]
    NUM_EPOCHS     = tr_cfg["num_epochs"]
    UNFREEZE_EPOCH = tr_cfg["unfreeze_epoch"]

    criterion = nn.CrossEntropyLoss(label_smoothing=loss_cfg.get("label_smoothing", 0.1))
    optimizer = optim.AdamW(model.classifier.parameters(),
                            lr=opt_cfg["lr_head"],
                            weight_decay=opt_cfg["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=sched_cfg.get("eta_min", 1e-6)
    )

    # ── Training loop ─────────────────────────────────────────────────
    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "lr", "epoch_time"]}
    best_val_acc  = 0.0
    best_model_wt = copy.deepcopy(model.state_dict())
    model_path    = cfg["output"]["model_path"]
    train_start   = time.time()

    print(f"\n{'Epoch':>5} | {'TrLoss':>7} | {'TrAcc':>6} | {'VaLoss':>7} | {'VaAcc':>6} | {'LR':>9} | {'Time':>6}")
    print("─" * 68)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        if epoch == UNFREEZE_EPOCH:
            print(f"\n🔓 Epoch {epoch}: Unfreeze backbone — fine-tune toàn bộ...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer.add_param_group({
                "params": model.features.parameters(),
                "lr": opt_cfg["lr_backbone"],
            })

        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        va_loss, va_acc = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        et = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        for k, v in zip(history.keys(), [tr_loss, va_loss, tr_acc, va_acc, lr, et]):
            history[k].append(v)

        flag = ""
        if va_acc > best_val_acc:
            best_val_acc  = va_acc
            best_model_wt = copy.deepcopy(model.state_dict())
            torch.save(best_model_wt, model_path)
            flag = " ✅"

        print(f"{epoch:>5} | {tr_loss:>7.4f} | {tr_acc:>5.1f}% | "
              f"{va_loss:>7.4f} | {va_acc:>5.1f}% | {lr:>9.2e} | {et:>5.1f}s{flag}")

    total_time = time.time() - train_start
    avg_epoch  = total_time / NUM_EPOCHS
    print(f"\n✅ Hoàn thành! Best Val Acc: {best_val_acc:.2f}%")
    print(f"   Thời gian: {total_time/60:.1f} phút | Mỗi epoch: {avg_epoch:.1f}s")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_training_curves(history, UNFREEZE_EPOCH, NUM_EPOCHS,
                         avg_epoch, best_val_acc, "/content/training_curves.png")

    model.load_state_dict(torch.load(model_path))
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            out = model(imgs.to(DEVICE))
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    plot_confusion_matrix(all_labels, all_preds, CLASS_NAMES, "/content/confusion_matrix.png")
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

    # ── Đóng gói kết quả ─────────────────────────────────────────────
    results_dir = cfg["output"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    output_files = [model_path, "/content/training_curves.png",
                    "/content/confusion_matrix.png"]
    for f in output_files:
        if os.path.exists(f):
            shutil.copy(f, results_dir)
    zip_path = cfg["output"]["zip_path"].replace(".zip", "")
    shutil.make_archive(zip_path, "zip", results_dir)
    print(f"\n📦 Đã nén kết quả → {zip_path}.zip")

    try:
        from google.colab import files
        files.download(f"{zip_path}.zip")
    except Exception:
        print(f"   Tải thủ công: Files panel → {zip_path}.zip → Download")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Đường dẫn tới file cấu hình YAML")
    args = parser.parse_args()
    main(args)