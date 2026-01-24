**Tổng Quan**
- Dự án phân loại lớp phủ đất nông nghiệp và ước lượng diện tích từ ảnh vệ tinh (Sentinel-2) bằng mô hình phân đoạn ảnh sâu học.
- Pipeline gồm: chuẩn bị dữ liệu (từ H5 → gộp tháng → xác thực pixel-perfect), huấn luyện mô hình, đánh giá và xuất báo cáo.

**Kiến Trúc Mô Hình**
- Mô hình chính: `GRAUpp` (Ghost Residual Attention UNet++), encoder dựa trên ResNet34, kết hợp khối GhostConv và attention (`ECA`/`CBAM`), hỗ trợ deep supervision.
- Định nghĩa trong [graupp.py](graupp.py). Mô hình xuất ra 5 lớp:
	- 0: Background
	- 1–4: Các mức lớp nông nghiệp (Level 1 đến Level 4)

**Thư Mục & Tệp Chính**
- [train.py](train.py): Huấn luyện mô hình, lưu kết quả tốt nhất, xuất confusion matrix, ảnh qualitative và lịch sử huấn luyện (Excel).
- [test_model.py](test_model.py): Chạy suy luận trên bộ kiểm thử, tính `Accuracy`, `IoU`, ước lượng tổng diện tích nông nghiệp (km²), lưu báo cáo Excel và ảnh minh hoạ.
- [preprocess_data.py](preprocess_data.py): Hàm tiền xử lý “pixel-perfect” và lớp `LandCoverDataset` dùng cho PyTorch `DataLoader`.
- [analyze_labels.py](analyze_labels.py): Phân tích chi tiết phân bố nhãn toàn bộ dataset, đề xuất class weights và vẽ biểu đồ tổng hợp.
- Tiện ích:
	- [utils/export_all_from_h5.py](utils/export_all_from_h5.py): Bung dữ liệu từ `dataset.h5` ra thư mục, chuyển 10 kênh → RGB, xuất mask one-hot → index.
	- [utils/merge_all_months.py](utils/merge_all_months.py): Gộp dữ liệu từ nhiều tháng về một cấu trúc chung cho `train/val/test`.
	- [utils/explore_dataset_universal.py](utils/explore_dataset_universal.py): Khảo sát tổng quan dataset (kích thước, dtype, giá trị mask) và trực quan vài mẫu.
- Yêu cầu phụ thuộc: [requirements.txt](requirements.txt)

**Cài Đặt Môi Trường**
Chạy trên Windows (khuyến nghị có GPU NVIDIA + CUDA).

1) Tạo môi trường Python 3.10+ và cài gói:
```
pip install -r requirements.txt
```

2) Kiểm tra PyTorch CUDA (tuỳ chọn):
```
python -c "import torch; print(torch.cuda.is_available())"
```

**Chuẩn Bị Dữ Liệu**
Quy trình tiêu chuẩn: H5 → gộp tháng → xác thực pixel-perfect.

1) Bung dữ liệu từ H5 (sửa đường dẫn H5/OUTPUT nếu cần) trong [utils/export_all_from_h5.py](utils/export_all_from_h5.py):
```
python utils/export_all_from_h5.py
```
- Kết quả: `DATASET_FULL_EXTRACTED/<split>/<month>/{images,masks}`

2) Gộp dữ liệu 11 tháng về một cấu trúc chung trong [utils/merge_all_months.py](utils/merge_all_months.py):
```
python utils/merge_all_months.py
```
- Kết quả: `DATASET_MERGED/train/images`, `DATASET_MERGED/train/masks` (tương tự cho `val`, `test`).

3) Xác thực và chuẩn hoá “pixel-perfect” để tạo bộ `dataset_validated` dùng cho huấn luyện.
Hàm nằm trong [preprocess_data.py](preprocess_data.py) — ví dụ gọi từ Python:
```
python - << "PY"
from preprocess_data import pixel_perfect_preprocess

img_dir = r"D:\git\agrivison\dataset\DATASET_MERGED\train\images"
mask_dir = r"D:\git\agrivison\dataset\DATASET_MERGED\train\masks"
output_dir = r"dataset_validated"

pixel_perfect_preprocess(img_dir, mask_dir, output_dir, target_size=128)
PY
```
- Kết quả: ảnh/mask đã chuẩn hoá, kèm các tệp phân chia `train.txt`, `val.txt`, `test.txt`.

4) Phân tích nhãn để đề xuất class weights (khuyến nghị thực hiện trước huấn luyện):
```
python analyze_labels.py --data-path D:\git\agrivison\dataset\DATASET_MERGED\train --output bao_cao_chi_tiet.png
```
- In ra weights gợi ý, lưu biểu đồ thống kê tổng hợp vào [bao_cao_chi_tiet.png](bao_cao_chi_tiet.png).

**Huấn Luyện**
Sử dụng [train.py](train.py). Tham số mặc định phù hợp cho ảnh 128×128.

```
python train.py --processed-path dataset_validated \
								--output outputs \
								--epochs 50 \
								--batch-size 64 \
								--lr 5e-4 \
								--patience 10 \
								--num-workers 2
```
- Lưu ý:
	- Loss: kết hợp CrossEntropy (bỏ qua `IGNORE_INDEX=255`) + DiceLoss.
	- Class weights: có sẵn giá trị mẫu; có thể thay bằng kết quả từ `analyze_labels.py`.
	- Tự động lưu `best_model.pth`, `confusion_matrix.png`, `qualitative.png` và `training_history.xlsx` trong thư mục `outputs/<timestamp>/`.

**Đánh Giá & Báo Cáo**
Chạy suy luận/báo cáo với [test_model.py](test_model.py):
```
python test_model.py --model-path outputs/<run>/best_model.pth \
										--test-dir D:\git\agrivison\dataset\DATASET_MERGED\test \
										--output outputs_test \
										--resolution 5.0
```
- Tính `Accuracy`, `mIoU`, diện tích nông nghiệp (km², gồm các lớp > 0) và lưu `Test_Report_Agri_Total.xlsx` + ảnh trực quan.
- Nếu không chỉ định `--model-path`, script cố gắng tự tìm mô hình mới nhất trong `outputs_landcover/run_*` (tuỳ phiên bản dữ liệu của bạn).

**Cấu Trúc Dự Án**
- Gốc dự án:
	- [analyze_labels.py](analyze_labels.py)
	- [graupp.py](graupp.py)
	- [preprocess_data.py](preprocess_data.py)
	- [train.py](train.py)
	- [test_model.py](test_model.py)
	- [requirements.txt](requirements.txt)
	- [run.sh](run.sh) (chạy trên Linux/macOS; Windows dùng lệnh `python` tương đương)
	- [utils/export_all_from_h5.py](utils/export_all_from_h5.py)
	- [utils/merge_all_months.py](utils/merge_all_months.py)
	- [utils/explore_dataset_universal.py](utils/explore_dataset_universal.py)

**Tham Số & Quy Ước Quan Trọng**
- Kích thước ảnh/mask chuẩn: 128×128 (có thể thay đổi qua `TARGET_SIZE`).
- Số lớp: 5 (0: Background; 1–4: các mức lớp nông nghiệp). Pixel có giá trị `5` sẽ được gộp về lớp `4` trong tiền xử lý.
- Chuẩn hoá ảnh: `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))`.
- Augmentation (train): lật ngang/dọc, xoay 90°, dịch/scale/rotate; luôn dùng `NEAREST` cho mask để đảm bảo tính toàn vẹn nhãn.
- `IGNORE_INDEX=255` dùng để loại bỏ pixel không hợp lệ khỏi loss/metrics.

**Mẹo & Lưu Ý**
- Đảm bảo ảnh và mask trùng tên gốc trước bước “pixel-perfect”; script sẽ tạo tên `*_valid` nhất quán.
- Với dữ liệu Sentinel-2 10 kênh, luôn chuyển về RGB uint8 trước khi lưu (`export_all_from_h5.py`) để tránh lỗi `CV_64F` khi dùng OpenCV.
- `analyze_labels.py` giúp cân bằng dữ liệu bằng cách tính class weights theo phân bố pixel thực tế.
- Nếu thay đổi kích thước ảnh, hãy chỉnh đồng bộ `TARGET_SIZE` và tham số `img_size` của `LandCoverDataset`.

**Lệnh Nhanh**
1) Khảo sát dataset tổng quan:
```
python utils/explore_dataset_universal.py
```
2) Phân tích nhãn chi tiết + weights:
```
python analyze_labels.py --data-path <path_to_train_or_masks> --output bao_cao_chi_tiet.png
```
3) Huấn luyện:
```
python train.py --processed-path dataset_validated --output outputs
```
4) Đánh giá & báo cáo:
```
python test_model.py --model-path <best_model.pth> --test-dir <dataset/test> --output outputs_test
```

**Báo Lỗi & Góp Ý**
- Nếu gặp vấn đề khi chạy `test_model.py` với `LandCoverDataset`, kiểm tra phiên bản lớp dataset và đường dẫn thư mục `images/masks` để đảm bảo tương thích với chữ ký hàm bạn đang dùng.
- Vui lòng tạo issue hoặc nêu rõ bước tái hiện để hỗ trợ nhanh chóng.

**Kiến Trúc GRAU++ (tóm tắt)**
```
Input (3x128x128)
	|
	Ghost-ResNet34 Encoder
	|-- x0_0 (low-level)
	|-- x1_0
	|-- x2_0
	|-- x3_0
	|-- x4_0 (high-level)
	|
	Decoder kiểu UNet++ (ResidualConv + Attention tuỳ chọn)
	|
	+-- x0_1: [x0_0, up(x1_0)] → ResidualConv(attention)
	+-- x1_1: [x1_0, up(x2_0)] → ResidualConv(attention)
	+-- x2_1: [x2_0, up(x3_0)] → ResidualConv(attention)
	+-- x3_1: [x3_0, up(x4_0)] → ResidualConv(attention)
	|
	+-- x0_2: [x0_0, x0_1, up(x1_1)] → ResidualConv(attention)
	+-- x1_2: [x1_0, x1_1, up(x2_1)] → ResidualConv(attention)
	+-- x2_2: [x2_0, x2_1, up(x3_1)] → ResidualConv(attention)
	|
	+-- x0_3: [x0_0, x0_1, x0_2, up(x1_2)] → ResidualConv(attention)
	+-- x1_3: [x1_0, x1_1, x1_2, up(x2_2)] → ResidualConv(attention)
	|
	+-- x0_4: [x0_0, x0_1, x0_2, x0_3, up(x1_3)] → ResidualConv(attention)
	|
	+-- Deep Supervision (khi training): outputs tại x0_1, x0_2, x0_3, x0_4
```

**Tài Liệu Tham Khảo**
- UNet: https://arxiv.org/abs/1505.04597
- UNet++: https://arxiv.org/abs/1807.10165
- ResNet: https://arxiv.org/abs/1512.03385
- GhostNet/Ghost Module: https://arxiv.org/abs/1911.11907
- CBAM (Attention): https://arxiv.org/abs/1807.06521
- ECA-Net (Efficient Channel Attention): https://arxiv.org/abs/1910.03151
- Dice Loss / V-Net: https://arxiv.org/abs/1606.04797
- IoU/Jaccard Index: https://en.wikipedia.org/wiki/Jaccard_index
- Albumentations (augmentation ảnh & mask): https://albumentations.ai/docs/
- OpenCV (xử lý ảnh): https://docs.opencv.org/
- HDF5 (định dạng dữ liệu lớn): https://portal.hdfgroup.org/display/HDF5/HDF5
- PyTorch `Dataset`/`DataLoader`: https://pytorch.org/docs/stable/data.html
- PyTorch AMP (mixed precision): https://pytorch.org/docs/stable/amp.html
- Sentinel‑2 (ESA): https://sentinel.esa.int/web/sentinel/missions/sentinel-2
- Review Deep Learning cho viễn thám (MDPI Remote Sensing): https://www.mdpi.com/2072-4292/12/14/2267
