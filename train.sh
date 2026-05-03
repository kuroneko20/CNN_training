#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# train.sh — Chạy pipeline huấn luyện đầy đủ trên Google Colab
#
# Cách dùng:
#   bash train.sh
# ──────────────────────────────────────────────────────────────────────

set -e  # Dừng ngay nếu có lỗi

echo "============================================================"
echo "  CNN Training — EfficientNet-B0"
echo "============================================================"

# ─── Bước 1: Cài đặt thư viện ─────────────────────────────────────
echo ""
echo "[1/4] Cài đặt thư viện..."
pip install -q gdown torchinfo matplotlib seaborn scikit-learn pyyaml

# ─── Bước 2: Tải & giải nén dataset từ Google Drive ───────────────
echo ""
echo "[2/4] Tải dataset từ Google Drive..."
python3 -c "
import gdown, zipfile, os, yaml

with open('configs/train.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

file_id  = cfg['dataset']['gdrive_file_id']
zip_path = cfg['dataset']['zip_path']
data_root = cfg['dataset']['data_root']

print(f'Downloading file ID: {file_id}')
gdown.download(f'https://drive.google.com/uc?id={file_id}', zip_path, quiet=False)

os.makedirs(data_root, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(data_root)
print('Dataset giải nén thành công.')
"

# ─── Bước 3: Phân tích dataset (EDA) ──────────────────────────────
echo ""
echo "[3/4] Phân tích dataset..."
python3 scripts/preprocess_data.py \
    --data_root /content/dataset \
    --output_dir /content

# ─── Bước 4: Huấn luyện ───────────────────────────────────────────
echo ""
echo "[4/4] Bắt đầu huấn luyện..."
python3 scripts/train.py --config configs/train.yaml

echo ""
echo "============================================================"
echo "  HOÀN THÀNH! Kết quả lưu tại /content/all_results.zip"
echo "============================================================"