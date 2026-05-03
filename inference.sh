#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# inference.sh — Chạy dự đoán trên ảnh mới
#
# Cách dùng:
#   bash inference.sh --input /content/test.jpg --class_names "cat,dog,bird"
#   bash inference.sh --input /content/test_folder/ --class_names "cat,dog,bird"
# ──────────────────────────────────────────────────────────────────────

set -e

echo "============================================================"
echo "  CNN Inference — EfficientNet-B0"
echo "============================================================"

# ─── Cài đặt thư viện ─────────────────────────────────────────────
echo ""
echo "[1/2] Kiểm tra thư viện..."
pip install -q torchinfo matplotlib seaborn pyyaml

# ─── Chạy inference ───────────────────────────────────────────────
echo ""
echo "[2/2] Bắt đầu inference..."
python3 scripts/inference.py \
    --config configs/inference.yaml \
    "$@"

echo ""
echo "============================================================"
echo "  HOÀN THÀNH!"
echo "============================================================"