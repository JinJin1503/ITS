**🚦 Nhận Diện Biển Báo Giao Thông Việt Nam (YOLOv5)**
Dự án này triển khai mô hình YOLOv5 để nhận diện biển báo giao thông trong môi trường ảnh, video và thời gian thực qua camera.

🔗 Nội dung
Tổng quan
Yêu cầu hệ thống
Cài đặt
Cấu trúc dự án
Sử dụng
Thông tin mô hình

💡 Tổng quan
Dự án sử dụng kiến trúc YOLOv5 (You Only Look Once, phiên bản 5) của Ultralytics để phát hiện và khoanh vùng các biển báo giao thông Việt Nam. Ứng dụng web được xây dựng bằng Flask cung cấp giao diện thân thiện cho phép người dùng:
1.Upload ảnh để nhận diện.
2.Upload video để xử lý hàng loạt.
3.Sử dụng webcam để nhận diện trực tiếp (Real-time).
Mô hình hiện đang hỗ trợ nhận diện 43 loại biển báo khác nhau, bao gồm biển cấm, biển báo nguy hiểm và biển hiệu lệnh...

🛠️ Yêu cầu hệ thống
Hệ điều hành: Windows (Vì code sử dụng đường dẫn font Windows và đường dẫn mô hình tuyệt đối D:/ITS/... ).
Python: 3.8+.
Thư viện: PyTorch (đã cài đặt phiên bản CPU torch-2.5.1+cpu ), OpenCV, Flask, PIL, NumPy, và các thư viện hỗ trợ YOLOv5.

⚙️ Cài đặt
Bước 1: Clone Repository
Sử dụng giao thức SSH đã thiết lập:
git clone git@github.com:JinJin1503/ITS.git
cd ITS
Bước 2: Chuẩn bị Môi trường Python
Bạn nên tạo một môi trường ảo để quản lý thư viện (ví dụ môi trường tên là yolov5env).
Tạo môi trường:
python -m venv yolov5env
source yolov5env/Scripts/activate  # Trên Windows
Bước 3: Cài đặt Thư viện
Tạo tệp requirements.txt bằng lệnh sau (để lấy danh sách thư viện môi trường) và cài đặt chúng:
Giả định bạn đã tạo file requirements.txt
pip install -r requirements.txt
Bước 4: Tải Trọng số Mô hình
Mô hình được cấu hình để tải tệp trọng số best_fixed.pt từ đường dẫn:
MODEL_PATH = "D:/ITS/yolov5/best_fixed.pt" 
Tải tệp best_fixed.pt của mô hình đã huấn luyện (Lưu ý: Tệp này không có trên GitHub do đã được loại trừ bằng .gitignore).
Đặt tệp này vào thư mục: D:/ITS/yolov5/

📁 Cấu trúc dự án
Đây là cấu trúc cơ bản cần thiết để chạy ứng dụng:
ITS/
├── app.py              # Logic chính của Flask app và YOLOv5 inference 
├── .gitignore          # Loại trừ môi trường ảo, file media và weights
├── yolov5/             # Thư mục chứa các module YOLOv5 (models, utils, etc.)
│   └── (code YOLOv5)
├── static/
│   ├── uploads/        # Nơi lưu trữ ảnh/video đầu vào
│   └── results/        # Nơi lưu trữ ảnh/video đã xử lý
├── templates/
   └── index.html      # Giao diện người dùng (Front-end)
🚀 Sử dụng
Bước 1: Khởi chạy Ứng dụng FlaskMở terminal trong thư mục gốc (ITS/) và chạy ứng dụng:Bashpython app.py
Ứng dụng sẽ chạy trên máy chủ cục bộ, thường là tại http://127.0.0.1:5000/.
Bước 2: Thao tác trên WebTruy cập địa chỉ trên trình duyệt và sử dụng các chức năng sau:Upload Ảnh/Video: Chọn tab tương ứng, tải file lên và nhấn "🔍 Phát Hiện Biển Báo".Camera Trực Tiếp: Nhấn "▶️ Bật Camera" để bắt đầu stream video từ webcam và nhận diện theo thời gian thực.Kết quả: Kết quả nhận diện (ảnh/video đã khoanh vùng) cùng với danh sách chi tiết các biển báo được phát hiện (mã hiệu, ý nghĩa và độ tin cậy) sẽ hiển thị ở khu vực kết quả.
ℹ️ Thông tin mô hình
Kiến trúc: YOLOv5 (v7.0-444-gdeec5e45) 
Framework: PyTorch (torch-2.5.1+cpu) 
Thiết bị: Chạy trên CPU (do cấu hình device=select_device('') và thông báo khởi động)
Các lớp nhận diện: Hỗ trợ 43 lớp, bao gồm các biển báo P (Cấm), R (Hiệu lệnh), W (Nguy hiểm) của Việt Nam.
