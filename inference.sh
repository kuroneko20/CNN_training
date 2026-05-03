#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# inference.sh — Chạy dự đoán trên ảnh mới
#
# Cách dùng:
#   !bash /content/CNN_training/inference.sh \
#       --input /content/test.jpg \
#       --weights /content/best_model.pth \
#       --class_names "classA,classB,classC"
# ──────────────────────────────────────────────────────────────────────

set -e

PROJECT="/content/CNN_training"
cd "$PROJECT"
echo "📂 Working directory: $(pwd)"

echo "============================================================"
echo "  CNN Inference — EfficientNet-B0"
echo "============================================================"

echo ""
echo "[1/2] Kiểm tra thư viện..."
pip install -q torchinfo matplotlib seaborn pyyaml

echo ""
echo "[2/2] Bắt đầu inference..."
python3 "$PROJECT/scripts/inference.py" \
    --config "$PROJECT/configs/inference.yaml" \
    "$@"

echo ""
echo "============================================================"
echo "  HOÀN THÀNH!"
echo "============================================================"