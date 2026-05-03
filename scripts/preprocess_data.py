"""
preprocess_data.py
──────────────────
Kiểm tra và tiền xử lý dataset trước khi train.
Phát hiện cấu trúc ImageFolder, thống kê phân bố lớp,
và sinh biểu đồ EDA lưu vào thư mục output.

Cách dùng (trên Google Colab):
    python scripts/preprocess_data.py --data_root /content/dataset
"""

import argparse
import os
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import warnings


# ─── Cấu hình ─────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─── Hàm tiện ích ─────────────────────────────────────────────────────

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
    """Tìm thư mục ImageFolder hợp lệ đầu tiên (parent_dir/class/image.jpg)."""
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
    """Tìm cặp (train_dir, val_dir) nếu có sẵn."""
    split_names = ["train", "Train", "training", "val", "Val",
                   "validation", "test", "Test", "testing"]
    found = {s.lower(): root / s for s in split_names if (root / s).exists()}
    train = found.get("train") or found.get("training")
    val = (found.get("val") or found.get("validation")
           or found.get("test") or found.get("testing"))
    return train, val


def verify_images(root: Path, class_stats: dict, max_check: int = 20):
    """Kiểm tra nhanh một số ảnh trong mỗi class để phát hiện file lỗi."""
    print("\n🔎 Kiểm tra tính hợp lệ của ảnh (tối đa", max_check, "ảnh/class):")
    broken = []
    for cls, count in class_stats.items():
        cls_path = root / cls
        imgs = sorted([f for f in cls_path.iterdir()
                       if f.suffix.lower() in IMG_EXTS])[:max_check]
        for img_path in imgs:
            try:
                safe_pil_loader(str(img_path))
            except Exception as e:
                broken.append((str(img_path), str(e)))

    if broken:
        print(f"  ⚠️  {len(broken)} ảnh lỗi:")
        for p, e in broken[:10]:
            print(f"     {p}: {e}")
    else:
        print(f"  ✅ Tất cả ảnh kiểm tra đều hợp lệ.")
    return broken


def plot_class_distribution(class_stats: dict, save_path: str):
    """Vẽ biểu đồ phân bố số lượng ảnh theo lớp."""
    classes = list(class_stats.keys())
    counts  = list(class_stats.values())
    total   = sum(counts)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    bars = ax.bar(range(len(classes)), counts, color="steelblue", edgecolor="white")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Số lượng ảnh")
    ax.set_title("Phân bố số lượng ảnh theo lớp", fontweight="bold")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    if all(c > 0 for c in counts):
        ax.pie(counts, labels=classes, autopct="%1.1f%%", startangle=90)
        ax.set_title("Tỉ lệ phần trăm các lớp", fontweight="bold")

    plt.suptitle(f"Tổng {total} ảnh / {len(classes)} lớp",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Đã lưu biểu đồ → {save_path}")


def plot_sample_images(root: Path, class_names: list, save_path: str):
    """Hiển thị ảnh mẫu từ mỗi lớp (tối đa 8 lớp)."""
    n_show = min(len(class_names), 8)
    fig, axes = plt.subplots(1, n_show, figsize=(3 * n_show, 3))
    if n_show == 1:
        axes = [axes]

    for i, cls in enumerate(class_names[:n_show]):
        cls_path = root / cls
        imgs_found = sorted([f for f in cls_path.iterdir()
                             if f.suffix.lower() in IMG_EXTS])
        if imgs_found:
            img = mpimg.imread(str(imgs_found[0]))
            axes[i].imshow(img)
            axes[i].set_title(f"{cls}\n({len(imgs_found)} ảnh)", fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Ảnh mẫu từ mỗi lớp", fontsize=13, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Đã lưu ảnh mẫu → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────

def main(args):
    data_root = Path(args.data_root)
    out_dir   = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  TIỀN XỬ LÝ & PHÂN TÍCH DATASET")
    print("=" * 60)

    # Tìm cấu trúc ImageFolder
    result = find_imagefolder_root(data_root)
    if result is None:
        raise RuntimeError("❌ Không tìm thấy cấu trúc ImageFolder hợp lệ.")

    found_root, class_stats = result
    class_names = list(class_stats.keys())
    num_classes = len(class_names)
    total_images = sum(class_stats.values())

    print(f"\n✅ ImageFolder tại: {found_root}")
    print(f"   Số lớp      : {num_classes}")
    print(f"   Tổng ảnh    : {total_images}")
    print(f"\n   Chi tiết:")
    for cls, cnt in class_stats.items():
        bar = "█" * (cnt * 20 // max(class_stats.values()))
        print(f"     {cls:20s}: {cnt:5d}  {bar}")

    # Kiểm tra train/val split
    train_split, val_split = find_splits(found_root.parent)
    if train_split is None:
        train_split, val_split = find_splits(found_root)

    if train_split and val_split and train_split != val_split:
        print(f"\n✅ Tìm thấy split: train={train_split.name}, val={val_split.name}")
    else:
        print(f"\nℹ️  Không có split sẵn → train.py sẽ tự chia stratified 80/20")

    # Verify ảnh
    broken = verify_images(found_root, class_stats)

    # Vẽ biểu đồ
    plot_class_distribution(class_stats, str(out_dir / "class_distribution.png"))
    plot_sample_images(found_root, class_names, str(out_dir / "sample_images.png"))

    print("\n" + "=" * 60)
    print("  KẾT QUẢ PHÂN TÍCH")
    print("=" * 60)
    print(f"  Số lớp      : {num_classes}")
    print(f"  Tổng ảnh    : {total_images}")
    print(f"  Ảnh lỗi     : {len(broken)}")
    print(f"  Biểu đồ     : {out_dir}/class_distribution.png")
    print(f"  Ảnh mẫu     : {out_dir}/sample_images.png")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess & EDA dataset")
    parser.add_argument("--data_root",  type=str, default="/content/dataset",
                        help="Thư mục gốc chứa dataset")
    parser.add_argument("--output_dir", type=str, default="/content",
                        help="Thư mục lưu biểu đồ đầu ra")
    args = parser.parse_args()
    main(args)