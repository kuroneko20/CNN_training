# 🧠 CNN Training — EfficientNet-B0 Fine-tuning

Dự án huấn luyện mô hình phân loại ảnh sử dụng **EfficientNet-B0** với chiến lược **2-stage fine-tuning** trên Google Colab.

---

## 📁 Cấu trúc dự án

```
CNN_training/
├── configs/
│   ├── train.yaml          # Toàn bộ hyperparameter cho training
│   └── inference.yaml      # Cấu hình dự đoán
├── sample_data/            # Thư mục chứa dữ liệu mẫu / test nhỏ
├── scripts/
│   ├── preprocess_data.py  # EDA & kiểm tra dataset
│   ├── train.py            # Script huấn luyện chính
│   └── inference.py        # Script dự đoán ảnh mới
├── inference.sh            # Chạy inference nhanh
├── train.sh                # Chạy toàn bộ pipeline train
├── requirements.txt        # Thư viện cần cài thêm
└── README.md
```

---

## ⚙️ Thông số mô hình

| Thông số | Giá trị |
|---|---|
| **Backbone** | EfficientNet-B0 |
| **Pre-trained weights** | ImageNet (IMAGENET1K_V1) |
| **Input size** | 224 × 224 px |
| **Classifier head** | 1280 → Dropout(0.3) → Linear(256) → ReLU → Dropout(0.2) → Linear(N_classes) |
| **Tổng tham số** | ~5.3M |
| **Framework** | PyTorch |

## ⚙️ Thông số huấn luyện

| Thông số | Giá trị |
|---|---|
| **Epochs** | 20 |
| **Batch size** | 64 |
| **Optimizer** | AdamW |
| **LR head (warm-up)** | 1e-3 |
| **LR backbone (fine-tune)** | 1e-5 |
| **Weight decay** | 1e-4 |
| **Scheduler** | CosineAnnealingLR (eta_min=1e-6) |
| **Loss** | CrossEntropyLoss (label_smoothing=0.1) |
| **Unfreeze epoch** | 5 |
| **Val split** | 20% (stratified, nếu không có sẵn split) |
| **Seed** | 42 |

## ⚙️ Thông số DataLoader

| Thông số | Giá trị | Lý do |
|---|---|---|
| `batch_size` | 64 | Tận dụng tối đa GPU T4 (15 GB VRAM) |
| `num_workers` | 4 | Prefetch nhanh hơn |
| `persistent_workers` | True | Giữ worker giữa epochs, giảm overhead ~10–20% |
| `prefetch_factor` | 2 | Chuẩn bị sẵn batch tiếp theo |
| `pin_memory` | True | Tăng tốc chuyển data CPU → GPU |

## 📊 Data Augmentation

| Kỹ thuật | Thông số |
|---|---|
| Resize | 256 × 256 |
| RandomCrop | 224 × 224 |
| RandomHorizontalFlip | p = 0.5 |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2 |
| RandomRotation | ±15° |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

## 🔁 Chiến lược Fine-tuning 2 Giai Đoạn

| Giai đoạn | Epoch | Backbone | LR Head | LR Backbone |
|---|---|---|---|---|
| 1 — Warm-up | 1 – 4 | ❄️ Đóng băng | 1e-3 | — |
| 2 — Fine-tune | 5 – 20 | 🔓 Mở khóa | decay cosine | 1e-5 |

---

## 🚀 Hướng dẫn sử dụng trên Google Colab

### Bước 1 — Clone repo

```bash
!git clone https://github.com/kuroneko20/CNN_training.git
```

> ⚠️ **Không cần `%cd CNN_training`** — các script tự tìm đúng thư mục.

### Bước 2 — Cài đặt thư viện

```bash
!pip install -r requirements.txt
```

### Bước 3 — Cấu hình dataset

> ✅ **Nếu dùng dataset `Landmark_Classification.zip` mặc định: không cần chỉnh gì.** File ID đã được điền sẵn trong `configs/train.yaml`.

Nếu muốn dùng dataset khác, mở `configs/train.yaml` và thay `gdrive_file_id`:

```yaml
dataset:
  gdrive_file_id: "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
```

Lấy File ID từ link Google Drive của bạn:
```
https://drive.google.com/file/d/<FILE_ID>/view
                                  ^^^^^^^^
                               copy phần này
```

> ⚠️ Đảm bảo file Google Drive được đặt chế độ **"Anyone with the link"** thì `gdown` mới tải được.

### Bước 4 — Chạy training

**Cách A — Chạy toàn bộ pipeline 1 lần (khuyên dùng):**

```bash
!bash /content/CNN_training/train.sh
```

**Cách B — Chạy từng bước thủ công:**

```python
# Cell 1 — Tải và giải nén dataset
import gdown, zipfile, os, yaml
with open('configs/train.yaml') as f:
    cfg = yaml.safe_load(f)
gdown.download(f'https://drive.google.com/uc?id={cfg["dataset"]["gdrive_file_id"]}',
               cfg['dataset']['zip_path'], quiet=False)
os.makedirs(cfg['dataset']['data_root'], exist_ok=True)
with zipfile.ZipFile(cfg['dataset']['zip_path']) as zf:
    zf.extractall(cfg['dataset']['data_root'])
print('Done!')
```

```bash
# Cell 2 — Phân tích dataset (EDA)
!python scripts/preprocess_data.py --data_root /content/dataset

# Cell 3 — Train
!python scripts/train.py --config configs/train.yaml
```

**Cách C — Dùng notebook có sẵn:**

Upload file `CNN_Training_Complete_v2_run.ipynb` lên Colab rồi chạy từng cell.

### Bước 5 — Kết quả

Sau khi train xong, file `all_results.zip` sẽ được tự động download. Bao gồm:

| File | Mô tả |
|---|---|
| `best_model.pth` | Trọng số model tốt nhất (theo val accuracy) |
| `training_curves.png` | Loss, Accuracy, LR schedule, thời gian/epoch |
| `confusion_matrix.png` | Ma trận nhầm lẫn trên tập validation |

---

## 🔍 Dự đoán ảnh mới (Inference)

Inference đã có sẵn trong project, **không cần viết thêm code**. Làm theo các bước sau:

### Bước 1 — Lấy danh sách class names

Sau khi `train.sh` chạy xong, script tự in ra danh sách lớp. Tìm dòng có dạng:

```
✅ Dataset: 5 lớp — ['HaLong', 'HoiAn', 'HoanKiem', 'MySon', 'PhongNha']
```

Copy danh sách đó, bỏ dấu ngoặc vuông và quotes, nối bằng dấu phẩy:
```
HaLong,HoiAn,HoanKiem,MySon,PhongNha
```

> Nếu không nhớ, chạy lệnh này để xem lại:
> ```python
> from torchvision import datasets
> ds = datasets.ImageFolder('/content/dataset')
> print(','.join(ds.classes))
> ```

### Bước 2 — Upload ảnh cần dự đoán lên Colab

Dùng Files panel bên trái → Upload, hoặc chạy trong notebook:

```python
from google.colab import files
uploaded = files.upload()   # chọn ảnh từ máy tính
```

### Bước 3 — Chạy inference

**Dự đoán 1 ảnh:**

```bash
!python scripts/inference.py \
    --config configs/inference.yaml \
    --weights /content/best_model.pth \
    --input /content/test.jpg \
    --class_names "HaLong,HoiAn,HoanKiem,MySon,PhongNha"
```

Output mẫu:
```
📷 Ảnh   : test.jpg
   Dự đoán: HoiAn
   Tin cậy: 94.32%

   Phân phối xác suất:
   HoiAn               : 94.32%  ██████████████████████████████
   HaLong              :  3.21%  █
   HoanKiem            :  1.47%
   MySon               :  0.72%
   PhongNha            :  0.28%
```

**Dự đoán cả thư mục (lưu kết quả ra CSV):**

```bash
!python scripts/inference.py \
    --config configs/inference.yaml \
    --weights /content/best_model.pth \
    --input /content/test_folder/ \
    --class_names "HaLong,HoiAn,HoanKiem,MySon,PhongNha"
```

Kết quả lưu tại `/content/predictions.csv`:
```
file,prediction,confidence
/content/test_folder/img1.jpg,HoiAn,0.9432
/content/test_folder/img2.jpg,HaLong,0.8811
...
```

**Hoặc dùng shell script:**

```bash
!bash /content/CNN_training/inference.sh \
    --input /content/test.jpg \
    --weights /content/best_model.pth \
    --class_names "HaLong,HoiAn,HoanKiem,MySon,PhongNha"
```

### Lưu ý

| Tình huống | Cách xử lý |
|---|---|
| Confidence < 50% | Model không chắc chắn — ảnh có thể không thuộc dataset |
| Muốn dùng model ở session khác | Upload lại `best_model.pth` từ `all_results.zip` đã download |
| Quên class names | Chạy lệnh xem classes ở Bước 1 |

---

## 📐 Cấu trúc dataset yêu cầu

Dataset phải ở định dạng **ImageFolder** của PyTorch:

```
dataset.zip
└── dataset/
    ├── class_A/
    │   ├── img001.jpg
    │   └── ...
    ├── class_B/
    │   └── ...
    └── class_C/
        └── ...
```

Hoặc nếu đã có sẵn train/val split:

```
dataset.zip
└── dataset/
    ├── train/
    │   ├── class_A/
    │   └── class_B/
    └── val/
        ├── class_A/
        └── class_B/
```

Script tự động phát hiện cấu trúc. Nếu không có split → tự chia **stratified 80/20**.

---

## 🛠️ Tuỳ chỉnh hyperparameter

Chỉnh sửa `configs/train.yaml` — không cần sửa code:

```yaml
training:
  num_epochs: 30          # Tăng số epoch
  unfreeze_epoch: 8       # Warm-up lâu hơn

dataloader:
  batch_size: 32          # Giảm nếu GPU ít VRAM

optimizer:
  lr_head: 5.0e-4         # Thay LR

loss:
  label_smoothing: 0.05   # Giảm label smoothing
```

---

## 📦 Thư viện sử dụng

| Thư viện | Mục đích |
|---|---|
| PyTorch + Torchvision | Deep learning framework |
| torchinfo | In tóm tắt kiến trúc model |
| gdown | Tải dataset từ Google Drive |
| scikit-learn | Confusion matrix, classification report |
| matplotlib + seaborn | Visualize kết quả |
| PyYAML | Đọc file cấu hình |

---

## 💡 Lý do chọn EfficientNet-B0

| Tiêu chí | EfficientNet-B0 | ResNet50 |
|---|---|---|
| Số tham số | ~5.3M | ~25M |
| Top-1 ImageNet Acc | 77.1% | 76.1% |
| FLOPs | 0.39B | 4.1B |
| Tốc độ (Colab T4) | ~30s/epoch | ~60s/epoch |

> EfficientNet-B0 đạt accuracy cao hơn ResNet50 với ít tham số hơn 5×, phù hợp để fine-tuning nhanh trên Colab.