import os
import shutil
import pandas as pd
from tqdm import tqdm

# =========================================================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================================================
# [QUAN TRỌNG] Thay đổi đường dẫn này trỏ đến folder MEGA_GRID của bạn
SOURCE_ROOT = r"E:\datdai\outputs_grid_search\MEGA_GRID_20260126_1519"

# Thư mục đích để chứa các file đã đổi tên
DEST_DIR = r"E:\datdai\grid_search_summary_files"


def main():
    if not os.path.exists(SOURCE_ROOT):
        print(f"[LỖI] Không tìm thấy thư mục nguồn: {SOURCE_ROOT}")
        return

    # Tạo thư mục đích nếu chưa có
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"[INFO] Đang quét file log trong: {SOURCE_ROOT}")
    print(f"[INFO] File sau khi đổi tên sẽ lưu tại: {DEST_DIR}")

    summary_data = []

    # Duyệt cây thư mục
    count = 0
    for root, dirs, files in os.walk(SOURCE_ROOT):
        for file in files:
            if file == "train_log.xlsx":
                # 1. Lấy tên Folder cha (chính là tên cấu hình: Adam_lr_...)
                config_name = os.path.basename(root)

                # Đường dẫn file gốc
                src_file_path = os.path.join(root, file)

                # 2. Tạo tên file mới: Adam_lr_0.0001_seed_0.xlsx
                new_filename = f"{config_name}.xlsx"
                dest_file_path = os.path.join(DEST_DIR, new_filename)

                # 3. Copy sang thư mục tổng hợp
                shutil.copy2(src_file_path, dest_file_path)

                # 4. (BONUS) Đọc file để lấy kết quả tốt nhất
                try:
                    df = pd.read_excel(src_file_path)
                    if not df.empty:
                        # Tìm dòng có mIoU cao nhất trong file này
                        best_row = df.loc[df['mIoU'].idxmax()]

                        # Thêm thông tin cấu hình vào dòng đó
                        log_entry = best_row.to_dict()
                        log_entry['Configuration'] = config_name

                        # Tách tên config để lấy Opt, LR, Seed riêng (để sort cho dễ)
                        # Giả định tên folder: Adam_lr_0.0001_seed_0
                        parts = config_name.split('_')
                        if len(parts) >= 4:
                            log_entry['Optimizer'] = parts[0]
                            log_entry['LR'] = float(parts[2])
                            log_entry['Seed'] = int(parts[4])

                        summary_data.append(log_entry)
                        count += 1
                except Exception as e:
                    print(f"Lỗi đọc file {new_filename}: {e}")

    print(f"\n[XONG] Đã copy và đổi tên {count} file log thành công!")

    # 5. Xuất file Master Ranking
    if summary_data:
        print("[INFO] Đang tạo file tổng sắp xếp hạng (MASTER_RANKING.xlsx)...")
        df_master = pd.DataFrame(summary_data)

        # Sắp xếp cột cho đẹp (Đưa tên Cấu hình lên đầu)
        cols = ['Configuration', 'Optimizer', 'LR', 'Seed', 'epoch', 'mIoU', 'accuracy', 'val_loss']
        # Lấy thêm các cột còn lại nếu có
        other_cols = [c for c in df_master.columns if c not in cols]
        df_master = df_master[cols + other_cols]

        # Sắp xếp theo mIoU giảm dần (Cao nhất lên đầu)
        df_master = df_master.sort_values(by='mIoU', ascending=False)

        master_path = os.path.join(DEST_DIR, "MASTER_RANKING.xlsx")

        # Lưu file Excel với định dạng màu mè chút cho dễ nhìn
        with pd.ExcelWriter(master_path, engine='xlsxwriter') as writer:
            df_master.to_excel(writer, index=False, sheet_name='Ranking')
            workbook = writer.book
            worksheet = writer.sheets['Ranking']

            # Highlight top 1 màu xanh lá
            green_fmt = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
            worksheet.set_row(1, None, green_fmt)  # Dòng đầu tiên (sau header)

            # Chỉnh độ rộng cột configuration
            worksheet.set_column('A:A', 30)

        print(f"[XUẤT SẮC] File bảng xếp hạng tổng hợp nằm tại: {master_path}")
        print(f"-> Top 1 hiện tại là: {df_master.iloc[0]['Configuration']} với mIoU: {df_master.iloc[0]['mIoU']:.4f}")


if __name__ == "__main__":
    main()