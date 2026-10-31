# Tác dụng các file trong dự án demo

## 1. Các file chính cần giữ để chạy giao diện

| File | Tác dụng |
|---|---|
| [gui.py](../gui.py) | File khởi động giao diện desktop bằng Tkinter. Tạo cửa sổ, nút chọn model, chọn ảnh/video/camera, Bắt đầu/Dừng, hiển thị khung hình và số lượng nhận diện. Tên file không bắt buộc phải là app.py. |
| [src/gui_engine.py](../src/gui_engine.py) | Bộ xử lý cho giao diện. Đọc ảnh hoặc từng khung hình video/camera, gọi model để nhận diện, gửi kết quả về giao diện, lưu ảnh/video và log. Hàm _run_images xử lý ảnh; _run_stream xử lý video/camera; process điều phối chế độ và tải model. Đây là nơi thay thế chức năng của các file detect_image.py, detect_video.py, detect_camera.py trước đây. |
| [src/common.py](../src/common.py) | Các công cụ dùng chung: cấu hình mặc định, tải model bằng YOLO, gọi model.predict, đọc hộp nhận diện và confidence, phân loại helmet/no_helmet, đếm đối tượng, vẽ kết quả và ghi CSV/báo cáo. |
| [src/__init__.py](../src/__init__.py) | Đánh dấu thư mục src là một gói Python. Hiện chỉ có mô tả, không chứa thuật toán nhận diện hay huấn luyện. Nên giữ cùng các file xử lý. |
| [models/best_v1.pt](../models/best_v1.pt) | Trọng số model đã huấn luyện. Chứa các tham số đã học để nhận diện; giao diện tải file này khi bắt đầu chạy. Demo không huấn luyện lại. Model hiện có hai nhãn helmet và no-helmet. |

## 2. Thư viện, dữ liệu mẫu và tài liệu

| File/thư mục | Tác dụng |
|---|---|
| [requirements.txt](../requirements.txt) | Danh sách thư viện cần cài: ultralytics cho YOLO, opencv-python cho ảnh/video/camera, numpy cho dữ liệu ảnh, Pillow cho hiển thị ảnh trên Tkinter. Tkinter đi kèm nhiều bản cài Python và không được cài bằng danh sách này. |
| [test_images/demo_helmet.jpg](./demo_helmet.jpg) | Ảnh đầu vào mẫu để thử chế độ ảnh. Không phải model hay mã train. Có thể thay bằng ảnh khác hoặc chọn ảnh ở thư mục bất kỳ. |
| [test_videos/demo.mp4](../test_videos/demo.mp4) | Video đầu vào mẫu để thử chế độ video. Chương trình đọc từng khung hình rồi nhận diện. Không phải dữ liệu bắt buộc phải dùng khi train. |
| [outputs/](../outputs/) | Nơi chương trình lưu ảnh kết quả, video kết quả, log CSV và báo cáo tổng kết. Được tạo khi chạy nếu chưa có. Số lượt phát hiện cộng qua nhiều khung hình không phải số người duy nhất. |
| [tests/test_common.py](../tests/test_common.py) | Kiểm thử một số chức năng mã nguồn dùng chung. Không cần để mở giao diện và không chứng minh độ chính xác của model. |
| [README.md](../README.md) | Tài liệu giới thiệu/hướng dẫn. Không tham gia thực thi. Hiện vẫn có hướng dẫn dòng lệnh cũ liên quan đến main.py đã được bỏ; để chạy bản giao diện hiện tại, dùng lệnh bên dưới. |
| [test_images/SOURCES.md](./SOURCES.md) | Chính tài liệu này: giải thích vai trò file và ghi nguồn dữ liệu mẫu. Không tham gia nhận diện, xóa tài liệu không làm thay đổi kết quả. |
| src/__pycache__/ | Bộ nhớ đệm mã Python tự sinh, có thể xuất hiện lại khi chạy. Không chứa kiến thức đã học của model; xóa thì Python tự tạo lại. |
| .venv/ | Môi trường Python riêng của dự án và các thư viện đã cài. Không phải mã giao diện hay model. Không nên xóa nếu đang dùng môi trường này để chạy demo. |

## 3. Luồng hoạt động và cách chạy

1. Chạy gui.py để mở giao diện.
2. Chọn file model đã train và ảnh/video hoặc camera.
3. Nhấn Bắt đầu: gui.py chuyển cấu hình cho gui_engine.py.
4. gui_engine.py dùng common.py tải trọng số và gọi dự đoán.
5. Kết quả được gửi về giao diện và lưu vào outputs/.

Chạy từ thư mục gốc dự án trên Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe gui.py
```

Không cần file train hay dataset train để chạy demo. Cần Python, các thư viện, model đã train và dữ liệu đầu vào. Việc chạy không báo lỗi không có nghĩa mọi nhận diện đều chính xác.

## 4. Nguồn ảnh/video mẫu

Nguồn công khai: https://github.com/SulfaSaji/HelmCheck (tải ngày 2026-10-28).

- demo_helmet.jpg: đường dẫn gốc `demo test dataset/images/BikesHelmets37_png.rf.b80a8db516c2f87565483d405f3991cf.jpg`.
- demo.mp4: đường dẫn gốc `demo test dataset/videos/video_helmet_no_helmet.mp4`.

Các mẫu được dùng để kiểm tra luồng phần mềm, không phải bộ đánh giá độ chính xác có nhãn chuẩn. Chưa xác minh chúng có trùng dữ liệu train của model hay không.
