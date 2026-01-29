# Agrivison — Phân Đoạn Nông Nghiệp & Ước Tính Diện Tích

Phân đoạn ảnh vệ tinh đa phổ (10 kênh) thành 5 lớp (0–4) và ước tính diện tích theo lớp (km²). Repo cung cấp pipeline gọn gàng: xử lý dữ liệu → huấn luyện → đánh giá → báo cáo.

## Mục Lục
- Tổng quan nhanh
- Sơ đồ pipeline
- Cấu trúc dữ liệu
- Bắt đầu nhanh
- Quy trình chi tiết
- Mô hình & Loss
- Khắc phục sự cố
- Sơ đồ repo

## Tổng Quan Nhanh
- Nhiệm vụ: phân đoạn 5 lớp (0–4), pixel 5m để tính km².
- Mô hình: GRAU++ (encoder ResNet34 + Ghost + Attention ECA/CBAM).
- Thành phần chính: [graupp.py](graupp.py), [preprocess_data.py](preprocess_data.py), [train.py](train.py), [test_model.py](test_model.py).

## Sơ Đồ Pipeline

```text
Pipeline

[HDF5 gốc: Sentinel-2 & labels]
    | export_all_from_h5.py
    v
[OUTPUT_ROOT: train/val/test per-month images,masks]
    | merge_all_months.py
    v
[dataset_final: train/val/test images,masks]
    | generate_split_txt.py
    v
[train.txt | val.txt | test.txt]
    | verify_data.py
    v
[Kiểm tra tồn tại & rò rỉ] --OK--> (tuỳ chọn) dataset_pruning.py -> rebalance_data.py (70/20/10)
    | normalize_split_txt.py
    v
[Split files chuẩn: basename only]
    | \
    |  \--> analyze_labels.py --> [Thống kê nhãn, gợi ý class weights]
    |
    +--> train.py --> [Huấn luyện: log + best model]
    |
    +--> gridsearch.py --> [Grid Search: Optimizer × LR × Seed]
            |
            +--> bestgridsearch.py --> [MASTER_RANKING.xlsx]
    |
    +--> test_model.py --> [Test report: IoU + Area km²]
```
Lưu ý:
- `OUTPUT_ROOT` là thư mục trung gian sau khi xuất từ HDF5.
- `dataset_final/` là thư mục dữ liệu cuối cùng dùng cho huấn luyện và đánh giá.
- Mọi script huấn luyện (`train.py`, `gridsearch.py`, `test_model.py`) đều đọc từ `dataset_final/`.
Gợi ý: Xem chi tiết từng bước ở phần “Quy Trình Chi Tiết” bên dưới.

## Cấu Trúc Dữ Liệu

Thư mục dữ liệu: `dataset_final/`

```
dataset_final/
	train/
		images/*.tif    # 10 kênh (H, W, 10), float32/64
		masks/*.png     # 1 kênh, uint8; nhãn 0–4, 255 = IGNORE
	val/
		images/*.tif
		masks/*.png
	test/
		images/*.tif
		masks/*.png
	train.txt        # Danh sách tên file (không kèm phần mở rộng), 1 tên/ dòng
	val.txt
	test.txt
```

Lưu ý quan trọng:
- Bộ đọc dữ liệu chính là `LandCoverDataset` trong [preprocess_data.py](preprocess_data.py).

 

## Bắt Đầu Nhanh

```bash
pip install -r requirements.txt
python utils/generate_split_txt.py --root dataset_final
python verify_data.py
python train.py --processed-path dataset_final --output outputs --epochs 100 --batch-size 64
python test_model.py --model-path outputs/FINAL_RUN_YYYYMMDD_HHMMSS/best_model.pth --test-dir dataset_final/test --output outputs_test
```

Nếu `.txt` đang ở dạng `name,source`, chuẩn hoá về `name`:

```bash
python utils/normalize_split_txt.py --root dataset_final
```

 

## Quy Trình Chi Tiết

1) Xuất dữ liệu đa phổ từ HDF5 (10 kênh TIF)
- Sửa đường dẫn trong [utils/export_all_from_h5.py](utils/export_all_from_h5.py): `H5_PATH`, `OUTPUT_ROOT`, `TARGET_LABEL_TYPE`.
- Chạy:

```bash
python utils/export_all_from_h5.py
```

Kết quả: `OUTPUT_ROOT/{train,val,test}/{month}/images/*.tif` và `.../masks/*.png` (mask từ one-hot về index 0–4).

2) Gộp dữ liệu 11 tháng về một nơi
- Sửa `SOURCE_ROOT`, `DEST_ROOT` trong [utils/merge_all_months.py](utils/merge_all_months.py).
- Chạy:

```bash
python utils/merge_all_months.py
```

Kết quả: `DEST_ROOT/{train,val,test}/images|masks` với tên file đã thêm prefix tháng.

3) Tạo danh sách `train.txt`, `val.txt`, `test.txt` (basename thuần)
- Dùng tiện ích Python để sinh danh sách từ thư mục ảnh:

```bash
python utils/generate_split_txt.py --root dataset_final
```

4) Kiểm tra tính toàn vẹn + rò rỉ dữ liệu
- Dùng [verify_data.py](verify_data.py) để kiểm tra sự tồn tại ảnh/mask và rò rỉ giữa các tập.

```bash
python verify_data.py
```

Ghi chú: Script này hỗ trợ định dạng dòng “name,source”. Nếu bạn đã tạo `.txt` thuần “name” ở bước (3), có thể bỏ qua phần `source` trong báo cáo; quan trọng là ảnh/mask vật lý tồn tại và không rò rỉ.

5) Tinh gọn dữ liệu (pruning) – tùy chọn
- Loại mẫu nền thuần hoặc ít thông tin bằng [dataset_pruning.py](dataset_pruning.py).

```bash
python dataset_pruning.py
```

Script sẽ phân tích tỷ lệ foreground, gợi ý di chuyển các cặp ảnh kém vào `dataset_final_TRASH/` (y/n).

6) Cân bằng & khử trùng lặp – tùy chọn
- [rebalance_data.py](rebalance_data.py) gom mẫu duy nhất và chia lại 70/20/10.

```bash
python rebalance_data.py
```

Chú ý: Script này xuất `.txt` theo dạng `name,source`. Nếu dùng `LandCoverDataset` mặc định, bạn hãy chuyển về danh sách tên thuần:

```bash
python utils/normalize_split_txt.py --root dataset_final
```

7) Phân tích nhãn & gợi ý trọng số lớp
- Dùng [analyze_labels.py](analyze_labels.py) để thống kê phân bố và đề xuất `class_weights`.

```bash
python analyze_labels.py --data-path dataset_final/train --output detailed_report.png
```

Xem đề xuất và cân nhắc cập nhật `FINAL_CLASS_WEIGHTS` trong [train.py](train.py).

8) Huấn luyện mô hình
- Tham số chính trong [train.py](train.py):
	- `--processed-path`: thư mục dữ liệu (mặc định: `dataset_final`).
	- `--output`: nơi lưu kết quả.
	- `--epochs`, `--batch-size`, `--num-workers`.

Chạy ví dụ:

```bash
python train.py --processed-path dataset_final --output outputs --epochs 100 --batch-size 64
```

Kết quả: tạo thư mục chạy `outputs/FINAL_RUN_YYYYMMDD_HHMMSS/` với `best_model.pth`, `train_log.xlsx`, biểu đồ học (`learning_curves.png`, `area_val_curve.png`) và ảnh minh hoạ.

9) Grid Search siêu tham số – 80 cấu hình
- Dùng [gridsearch.py](gridsearch.py) để thử các tổ hợp Optimizer × LR × Seed.

Khuyến nghị override đường dẫn dữ liệu về gốc `dataset_final` (vì script mặc định `--processed-path dataset_final/train`):

```bash
python gridsearch.py --processed-path dataset_final --output outputs_grid_search --epochs 15 --batch-size 32
```

Tổng hợp kết quả vào Excel: `.../MEGA_GRID_*/Grid_Search_Summary.xlsx`.

10) Gom log Grid Search về 1 nơi – tiện xem hạng
- Dùng [bestgridsearch.py](bestgridsearch.py). Sửa `SOURCE_ROOT`, `DEST_DIR` rồi chạy:

```bash
python bestgridsearch.py
```

Script sẽ copy các `train_log.xlsx` theo tên cấu hình và tạo `MASTER_RANKING.xlsx` đã sắp hạng theo mIoU.

11) Kiểm thử (Test) và báo cáo
- Dùng [test_model.py](test_model.py) để đánh giá trên `dataset_final/test`.

```bash
python test_model.py --model-path outputs/FINAL_RUN_YYYYMMDD_HHMMSS/best_model.pth --test-dir dataset_final/test --output outputs_test
```

Kết quả: `outputs_test/test_run_*/` chứa `Test_Report_Detailed.xlsx`, thống kê toàn cục (Accuracy, mIoU, IoU từng lớp, Area GT/Pred/Diff) và tối đa 50 ảnh minh hoạ (Input/GT/Pred/Overlay).

 

## Tính Diện Tích
- Kích thước pixel: 5m ⇒ diện tích mỗi pixel: $25\,m^2 = 2.5\times10^{-5}\,km^2$.
- Tổng diện tích theo lớp được cộng dồn theo mask dự đoán/nhãn, bỏ qua `255` (IGNORE).

 

## Khắc Phục Sự Cố
- `.txt` bị sai định dạng: Nếu dòng có `name,source`, hãy chuẩn hoá về tên thuần như bước (6).
- OOM (hết VRAM): giảm `--batch-size`, hoặc tắt deep supervision bằng sửa tham số khi khởi tạo model (nếu cần).
- Albumentations/torchvision lỗi import: kiểm tra phiên bản trong [requirements.txt](requirements.txt) và cài lại.
- Grid Search không thấy dữ liệu: đảm bảo `--processed-path` trỏ đến thư mục gốc chứa `train/ val/ test/`.

 

## Sơ Đồ Repo
- [train.py](train.py): huấn luyện + log + đồ thị + lưu model tốt nhất.
- [test_model.py](test_model.py): đánh giá, báo cáo chi tiết và ảnh minh hoạ.
- [gridsearch.py](gridsearch.py), [bestgridsearch.py](bestgridsearch.py): tìm siêu tham số và tổng hợp bảng xếp hạng.
- [preprocess_data.py](preprocess_data.py): `LandCoverDataset`, kiểm tra cấu trúc dữ liệu.
- [graupp.py](graupp.py): kiến trúc GRAU++.
- [dataset_pruning.py](dataset_pruning.py), [rebalance_data.py](rebalance_data.py), [verify_data.py](verify_data.py): tiện ích dữ liệu.
- [utils/](utils): xuất từ HDF5, gộp 11 tháng, khám phá nhanh bộ dữ liệu.

---
