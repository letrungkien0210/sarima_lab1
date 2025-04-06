import os
import pandas as pd
import glob # Thư viện glob để tìm file dễ dàng hơn

# --- Cấu hình ---
# Đường dẫn đến thư mục chứa các file CSV cần gộp
input_folder_path = 'data/meter_usages_1743839212343'

# Tên của file CSV sau khi gộp
output_file_name = 'merged_and_sorted_meter_usages.csv' # Đổi tên file đầu ra cho rõ ràng

# Tên cột dùng để sắp xếp
sort_column_name = "Last update"
# ----------------

# Kiểm tra xem thư mục đầu vào có tồn tại không
if not os.path.isdir(input_folder_path):
    print(f"Lỗi: Thư mục '{input_folder_path}' không tồn tại.")
else:
    # Tạo đường dẫn đầy đủ đến các file CSV trong thư mục
    all_csv_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

    if not all_csv_files:
        print(f"Không tìm thấy file CSV nào trong thư mục '{input_folder_path}'.")
    else:
        print(f"Tìm thấy {len(all_csv_files)} file CSV để gộp.")

        # Khởi tạo một danh sách rỗng để chứa các DataFrame
        list_of_dataframes = []

        # Đọc từng file CSV và thêm DataFrame vào danh sách
        for f in all_csv_files:
            try:
                df = pd.read_csv(f, low_memory=False)
                list_of_dataframes.append(df)
                print(f"Đã đọc: {f}")
            except Exception as e:
                print(f"Lỗi khi đọc file {f}: {e}")

        # Kiểm tra xem có DataFrame nào được đọc thành công không
        if not list_of_dataframes:
            print("Không có dữ liệu nào được đọc thành công. File gộp không được tạo.")
        else:
            # Gộp tất cả các DataFrame thành một DataFrame duy nhất
            merged_df = pd.concat(list_of_dataframes, ignore_index=True)
            print("\nĐang gộp các file...")

            # --- BƯỚC SẮP XẾP DỮ LIỆU ---
            # Kiểm tra xem cột cần sắp xếp có tồn tại trong DataFrame không
            if sort_column_name in merged_df.columns:
                print(f"Đang sắp xếp dữ liệu theo cột '{sort_column_name}' tăng dần...")
                try:
                    # Cố gắng chuyển đổi cột sang kiểu datetime để sắp xếp chính xác
                    # errors='coerce' sẽ biến các giá trị không phải ngày giờ hợp lệ thành NaT (Not a Time)
                    merged_df[sort_column_name] = pd.to_datetime(merged_df[sort_column_name], errors='coerce')

                    # Sắp xếp DataFrame tăng dần theo cột đã chỉ định
                    # na_position='last' sẽ đưa các hàng có giá trị NaT (không hợp lệ/trống) xuống cuối
                    merged_df = merged_df.sort_values(by=sort_column_name, ascending=True, na_position='last')
                    print("Sắp xếp hoàn tất.")
                except Exception as e:
                    print(f"Lỗi trong quá trình chuyển đổi hoặc sắp xếp cột '{sort_column_name}': {e}")
                    print("Dữ liệu sẽ được ghi ra file mà không được sắp xếp theo cột này.")
            else:
                print(f"Cảnh báo: Không tìm thấy cột '{sort_column_name}' trong dữ liệu đã gộp. Dữ liệu sẽ không được sắp xếp.")
            # --- KẾT THÚC BƯỚC SẮP XẾP ---

            # Lưu DataFrame đã gộp và sắp xếp thành file CSV mới
            try:
                merged_df.to_csv(output_file_name, index=False, encoding='utf-8-sig')
                print(f"\nHoàn thành! Đã gộp và sắp xếp dữ liệu vào file '{output_file_name}'.")
            except Exception as e:
                print(f"Lỗi khi lưu file gộp {output_file_name}: {e}")