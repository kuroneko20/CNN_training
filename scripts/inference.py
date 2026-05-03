"""
inference.py
────────────
Dự đoán nhãn cho ảnh mới sử dụng mô hình đã train.
Hỗ trợ cả single image và batch folder.

Cách dùng (trên Google Colab):
    # Dự đoán 1 ảnh
    python scripts/inference.py --config configs/inference.yaml \
        --input /content/test.jpg

    # Dự đoán cả thư mục
    python scripts/inference.py --config configs/inference.yaml \
        --input /content/test_images/
"""

import argparse
import csv
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import models, transforms


# ─── Hằng số ──────────────────────────────────────────────────────────
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ─── Helpers ──────────────────────────────────────────────────────────

def safe_pil_loader(path: str) -> Image.Image:
    """Load ảnh và convert sang RGB, xử lý đúng palette PNG có transparency."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            img = Image.open(f)
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            return img.convert("RGB")


def build_model_for_inference(num_classes: int, hidden: int,
                               dropout1: float, dropout2: float,
                               weights_path: str, device: str):
    """Tải EfficientNet-B0 và load checkpoint."""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout1, inplace=True),
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout2),
        nn.Linear(hidden, num_classes),
    )
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"✅ Đã load model từ: {weights_path}")
    return model


def get_transform(img_size: int, mean: list, std: list):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def predict_single(model, img_path: str, transform, device: str,
                   class_names: list, confidence_threshold: float):
    """Dự đoán cho 1 ảnh, trả về (class_name, confidence, all_probs)."""
    img = safe_pil_loader(img_path)
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1)[0]
    conf, idx = probs.max(0)
    pred_class = class_names[idx.item()]
    confidence = conf.item()
    if confidence < confidence_threshold:
        print(f"  ⚠️  Confidence thấp ({confidence:.2%}) — kết quả có thể không chính xác.")
    return pred_class, confidence, probs.cpu().numpy()


def predict_folder(model, folder_path: str, transform, device: str,
                   class_names: list, cfg: dict):
    """Dự đoán batch cho toàn bộ ảnh trong folder, lưu CSV."""
    folder   = Path(folder_path)
    img_paths = sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not img_paths:
        print(f"⚠️  Không tìm thấy ảnh trong: {folder_path}")
        return []

    results = []
    print(f"\nDự đoán {len(img_paths)} ảnh...")
    for i, img_path in enumerate(img_paths):
        pred, conf, _ = predict_single(
            model, str(img_path), transform, device,
            class_names, cfg["inference"]["confidence_threshold"]
        )
        results.append({"file": str(img_path), "prediction": pred, "confidence": f"{conf:.4f}"})
        if (i + 1) % 10 == 0 or i == len(img_paths) - 1:
            print(f"  [{i+1}/{len(img_paths)}] {img_path.name} → {pred} ({conf:.2%})")

    # Lưu CSV
    output_csv = cfg["inference"]["output_csv"]
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\n✅ Đã lưu kết quả → {output_csv}")
    return results


def visualize_predictions(model, img_paths, transform, device: str,
                           class_names: list, mean: list, std: list,
                           n_show: int = 16, save_path: str = "/content/predictions_sample.png"):
    """Hiển thị grid ảnh với nhãn dự đoán."""
    n = min(n_show, len(img_paths))
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    def denorm(t):
        m = torch.tensor(mean).view(3, 1, 1)
        s = torch.tensor(std).view(3, 1, 1)
        return (t * s + m).clamp(0, 1)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten() if nrows > 1 else [axes] * ncols

    for i, img_path in enumerate(img_paths[:n]):
        pred, conf, _ = predict_single(model, str(img_path), transform,
                                       device, class_names, 0.0)
        img_tensor = transform(safe_pil_loader(str(img_path)))
        img_np     = denorm(img_tensor).permute(1, 2, 0).numpy()
        axes[i].imshow(img_np)
        axes[i].set_title(f"{pred}\n({conf:.0%})", fontsize=9)
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    plt.suptitle("Dự đoán mẫu", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"✅ Đã lưu ảnh mẫu → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────

def main(args):
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {DEVICE.upper()}")

    # Class names
    if args.class_names:
        class_names = args.class_names.split(",")
    elif args.class_file:
        with open(args.class_file, "r") as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        raise ValueError(
            "Cần truyền --class_names 'cat,dog,bird' hoặc --class_file classes.txt"
        )
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    # Build model
    model_cfg = cfg["model"]
    model = build_model_for_inference(
        num_classes    = num_classes,
        hidden         = model_cfg.get("classifier_hidden", 256),
        dropout1       = model_cfg.get("dropout1", 0.3),
        dropout2       = model_cfg.get("dropout2", 0.2),
        weights_path   = args.weights or model_cfg["weights_path"],
        device         = DEVICE,
    )

    pre = cfg["preprocessing"]
    transform = get_transform(pre["img_size"], pre["imagenet_mean"], pre["imagenet_std"])

    input_path = Path(args.input or cfg["inference"]["input_path"])

    if input_path.is_file():
        # Single image
        pred, conf, probs = predict_single(
            model, str(input_path), transform, DEVICE,
            class_names, cfg["inference"]["confidence_threshold"]
        )
        print(f"\n📷 Ảnh   : {input_path.name}")
        print(f"   Dự đoán: {pred}")
        print(f"   Tin cậy: {conf:.2%}")
        print(f"\n   Phân phối xác suất:")
        for cls, p in sorted(zip(class_names, probs), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"   {cls:20s}: {p:6.2%}  {bar}")

    elif input_path.is_dir():
        # Batch folder
        results = predict_folder(model, str(input_path), transform, DEVICE, class_names, cfg)

        # Visualize sample
        img_paths = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in IMG_EXTS])
        if img_paths:
            visualize_predictions(
                model, img_paths, transform, DEVICE, class_names,
                pre["imagenet_mean"], pre["imagenet_std"],
                n_show=cfg["inference"].get("show_samples", 16),
            )
    else:
        print(f"❌ Không tìm thấy: {input_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference EfficientNet-B0")
    parser.add_argument("--config",      type=str, default="configs/inference.yaml")
    parser.add_argument("--input",       type=str, default=None,
                        help="Ảnh hoặc thư mục cần dự đoán")
    parser.add_argument("--weights",     type=str, default=None,
                        help="Override đường dẫn model .pth")
    parser.add_argument("--class_names", type=str, default=None,
                        help="Tên lớp phân cách bằng dấu phẩy: 'cat,dog,bird'")
    parser.add_argument("--class_file",  type=str, default=None,
                        help="File txt chứa tên lớp (1 lớp/dòng)")
    args = parser.parse_args()
    main(args)