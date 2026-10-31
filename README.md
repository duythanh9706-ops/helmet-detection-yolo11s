# DEMO YOLO11s – PHÁT HIỆN HELMET / NO_HELMET

Bộ mã nguồn demo cho bài tập lớn **Phát hiện người điều khiển xe máy không đội mũ bảo hiểm sử dụng Computer Vision**, bám theo kế hoạch nhóm và dựa trên hai file `detect_image.py`, `detect_video.py` đã cung cấp.

> **Lưu ý:** Bộ ZIP **chưa có trọng số đã huấn luyện** (`models/best_v1.pt`) và **chưa có ảnh/video kiểm thử** vì các tệp đó không được cung cấp. Không thể xác nhận độ chính xác, FPS hoặc kết quả dự đoán thực tế trước khi thêm model và dữ liệu.

## 1. Cấu trúc thư mục

```text
Helmet_NoHelmet_Demo/
├── main.py                  # Menu và giao diện lệnh chung
├── requirements.txt         # Thư viện cần thiết
├── run_demo.bat             # Nhấp đúp trên Windows để mở menu
├── run_demo.sh              # Mở menu trên macOS/Linux
├── models/
│   └── README.txt           # Đặt best_v1.pt tại đây
├── src/
│   ├── __init__.py
│   ├── common.py             # Model loader, config, vẽ nhãn, thống kê, log
│   ├── detect_image.py       # Nhận diện một hoặc nhiều ảnh
│   ├── detect_video.py       # Nhận diện từng frame của video
│   └── detect_camera.py      # Camera realtime, hiển thị FPS
├── test_images/             # Đặt ảnh đầu vào
├── test_videos/             # Đặt video đầu vào
├── outputs/                 # Ảnh/video kết quả, log
├── tests/                   # Kiểm tra mã nguồn không cần model
└── original/                # Bản gốc detect_image.py / detect_video.py
```

## 2. Cài đặt môi trường

Khuyến nghị Python **3.10–3.12**, sử dụng môi trường ảo, trên máy có webcam nếu cần demo realtime. Trong terminal tại thư mục dự án:

**Windows:**

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ultralytics thường cài kèm PyTorch làm phụ thuộc. Nếu máy dùng GPU CUDA, cài đặt đúng bộ PyTorch tương thích thiết bị theo hướng dẫn chính thức của PyTorch; nếu không, có thể demo với CPU (FPS thường thấp hơn).

## 3. Chuẩn bị model và dữ liệu

1. Chép **model YOLO11s đã huấn luyện 2 lớp Helmet / No Helmet** vào `models/best_v1.pt`.
2. Chép một số ảnh thử vào `test_images/` (ví dụ `test_images/demo.jpg`).
3. Chép một video thử vào `test_videos/` (ví dụ `test_videos/demo.mp4`).
4. Chạy từ **thư mục gốc** `Helmet_NoHelmet_Demo/`.

Kiểm tra phụ thuộc/model:

```bash
python main.py --check
```

Nếu model mang tên `best.pt`, chỉ cần giữ nguyên tên và dùng `--model models/best.pt`.

**Lưu ý về class:** chương trình dựa trên tên nhãn của mô hình như `helmet`, `no_helmet` (có hỗ trợ `No-Helmet`, `without helmet`). Nếu file model đặt tên khác, hệ thống vẫn có thể vẽ box nhưng sẽ đưa lớp đó vào `Other class`; cần chuẩn hóa nhãn trước khi báo cáo số lượng.

## 4. Chạy demo (cách dễ nhất)

```bash
python main.py --menu
```

Hoặc nhấp đúp `run_demo.bat` trên Windows. Chọn:

- **1 – Ảnh:** nhập `test_images` hoặc đường dẫn một ảnh; chọn `--show` tương đương hiển thị lần lượt ảnh kết quả. Nhấn **q** hoặc **Esc** để dừng xem; đối với một ảnh, nhấn phím bất kỳ để tiếp tục.
- **2 – Video:** nhập `test_videos/demo.mp4`; xem video từng frame, nhấn **q / Esc** để dừng.
- **3 – Camera realtime:** chọn camera ID `0`, giữ webcam hoạt động đến khi nhấn **q / Esc**. Có lựa chọn ghi lại video.
- **4 – Kiểm tra:** xem thư viện và model có sẵn không.

**Các lệnh nhanh:**

```bash
# Demo ảnh / thư mục ảnh
python main.py --mode image --input test_images --show

# Demo một ảnh
python main.py --mode image --input test_images/demo.jpg --output outputs/images --show

# Demo video và lưu MP4
python main.py --mode video --input test_videos/demo.mp4 --show

# Demo camera (nhấn q/ESC để dừng)
python main.py --mode camera --camera 0

# Demo camera và ghi video
python main.py --mode camera --camera 0 --record

# Model nằm ở đường dẫn khác / đổi confidence
python main.py --mode image --input test_images --model models/best.pt --conf 0.40 --show

# Thử giới hạn 100 frame video, chạy không hiện cửa sổ
python main.py --mode video --input test_videos/demo.mp4 --max-frames 100
```

## 5. Các tham số thống nhất với mã gốc

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---:|---|
| `--model` | `models/best_v1.pt` | Đường dẫn trọng số tùy biến |
| `--conf` | `0.32` | Ngưỡng confidence |
| `--iou` | `0.60` | Ngưỡng IoU dùng cho NMS |
| `--imgsz` | `640` | Kích thước ảnh đưa vào suy luận |
| `--camera` | `0` | ID camera |
| `--log-dir` | `outputs/logs` | Nơi lưu log |

Hai file gốc cũng dùng `agnostic_nms=False`. Chương trình mới giữ cấu hình này. Bạn có thể dùng `--device cpu` khi cần buộc chạy CPU.

## 6. Kết quả được tạo

- **Ảnh:** `outputs/images/<tên_ảnh>_result.jpg` có bbox, class/confidence và bảng đếm.
- **Video:** `outputs/videos/result_video.mp4` có bbox, nhãn, confidence, số lượng và FPS xử lý xấp xỉ.
- **Camera:** hiển thị trên cửa sổ OpenCV; nếu chọn `--record` sẽ ghi `outputs/camera/camera_result.mp4`.
- **Log:** `outputs/logs/<mode>_detections_<mốc_thời_gian>.csv` có các trường `timestamp`, `mode`, `source`, `frame_index`, `class_name`, `confidence`, `x1`, `y1`, `x2`, `y2`. File `*_summary_*.txt` lưu tổng số lượt phát hiện, số frame và thời gian xử lý.

**Không được diễn giải `total helmet/no_helmet` trên video là số người duy nhất:** cùng một người có thể xuất hiện qua nhiều frame. Chỉ số FPS là tốc độ xử lý xấp xỉ tại máy demo, không phải FPS chuẩn của mô hình trên mọi thiết bị.

## 7. Quy trình trình diễn trước lớp (3–5 phút)

1. Giới thiệu bài toán, mô hình YOLO11s và hai lớp `helmet / no_helmet`.
2. Mở `python main.py --check` để trình bày model và các thư viện đã sẵn sàng.
3. Mở menu **1 – Ảnh**, demo một ảnh có mũ và một ảnh không mũ. Nêu bbox, class, confidence, số lượng từng lớp.
4. Mở **2 – Video**, demo nhận diện frame-by-frame và video kết quả lưu lại.
5. Mở **3 – Camera**, cho thấy nhận diện realtime và FPS (nếu webcam và điều kiện môi trường phù hợp).
6. Mở thư mục `outputs/` và `outputs/logs/` để minh họa kết quả, phục vụ phần kiểm thử của Tuấn.
7. Kết luận ngắn về hạn chế: góc nghiêng, che khuất, thiếu sáng, vật thể xa có thể ảnh hưởng kết quả; chỉ đánh giá sau khi thử thực tế.

Chi tiết lời dẫn và checklist ở `KICH_BAN_DEMO.md`.

## 8. Kiểm thử mã nguồn không cần model

```bash
python -m unittest discover -s tests -v
```

Các bài kiểm tra này **chỉ xác minh cấu hình, chuẩn hóa tên class, lọc ảnh và ghi CSV**, không phải đánh giá độ chính xác YOLO11s. Muốn kiểm thử phát hiện thật phải cung cấp trọng số và ảnh/video kiểm thử.

## 9. Xử lý lỗi thường gặp

| Tình huống | Giải pháp |
|---|---|
| `Chưa có trọng số mô hình` | Chép `best_v1.pt` vào `models/` hoặc truyền `--model` |
| `No module named ultralytics` / `cv2` | Chạy `python -m pip install -r requirements.txt` trong đúng môi trường Python |
| `Không tìm thấy video / ảnh` | Kiểm tra file có tồn tại, nhập đường dẫn đúng và khởi chạy từ thư mục gốc |
| Không mở được webcam | Đóng Zoom/ứng dụng khác, cấp quyền camera, thử `--camera 1` |
| `imshow` không hoạt động | Chạy trên máy desktop có giao diện đồ họa; bỏ `--show` cho ảnh/video |
| Video MP4 không phát | Kiểm tra codec `mp4v` của OpenCV và trình phát; trên máy khác có thể cần chuyển mã |
| FPS thấp | Demo ở độ phân giải phù hợp, dùng GPU tương thích nếu có; ghi đúng điều kiện đo |
| `Other class` thay vì `No Helmet` | Kiểm tra `result.names` từ file model; tên class có thể khác |

## 10. Phạm vi và tính trung thực

Phần mềm chỉ thực hiện **phát hiện hộp giới hạn và gắn nhãn**, không có nhận diện biển số, OCR, website, tính phạt hay định danh từng người. Các tính năng nêu trong README là mô tả **thiết kế/mã nguồn**; chưa có minh chứng đầu ra thật khi thiếu model và bộ dữ liệu thử nghiệm.
