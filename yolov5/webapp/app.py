from flask import Flask, render_template, request, Response
import torch
import cv2
import os
import sys
from werkzeug.utils import secure_filename
from PIL import ImageFont, ImageDraw, Image
import numpy as np

app = Flask(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from utils.augmentations import letterbox

device = select_device('')  # CPU
MODEL_PATH = "D:/ITS/yolov5/best_fixed.pt"
model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False, fp16=False)
stride, names, pt = model.stride, model.names, model.pt

vietnamese_names = {
    0: "P.101 - Đường cấm",
    1: "P.102 - Cấm đi ngược chiều",
    2: "P.103a - Cấm xe ô tô",
    3: "P.103b - Cấm xe ô tô rẽ phải",
    4: "P.103c - Cấm xe ô tô rẽ trái",
    5: "P.104 - Cấm xe máy",
    6: "P.106a - Cấm xe ô tô tải",
    7: "P.106b - Cấm xe tải vượt quá trọng tải cho phép",
    8: "P.107 - Cấm xe ô tô khách và xe tải",
    9: "P.107a - Cấm xe ô tô khách",
    10: "P.107b - Cấm xe ô tô taxi",
    11: "P.110a - Cấm xe đạp",
    12: "P.111a - Cấm xe gắn máy",
    13: "P.111d - Cấm xe ba bánh không động cơ",
    14: "P.112 - Cấm người đi bộ",
    15: "P.115 - Hạn chế trọng tải toàn bộ xe",
    16: "P.117 - Hạn chế chiều cao xe",
    17: "P.123a - Cấm rẽ trái",
    18: "P.123b - Cấm rẽ phải",
    19: "P.124a - Cấm quay đầu",
    20: "P.124b - Cấm rẽ phải",
    21: "P.124b1 - Cấm ô tô quay đầu",
    22: "P.124c - Cấm rẽ trái và quay đầu",
    23: "P.125 - Cấm vượt",
    24: "P.127 - Tốc độ tối đa cho phép",
    25: "P.128 - Cấm bóp còi",
    26: "P.130 - Cấm dừng và đỗ xe",
    27: "P.131a - Cấm đỗ xe",
    28: "P.137 - Cấm rẽ trái và rẽ phải",
    29: "R.122 - Dừng lại",
    30: "R.302a - Hướng đi vòng chướng ngại vật",
    31: "W.201 - Chỗ ngoặt nguy hiểm",
    32: "W.202 - Nhiều chỗ ngoặt nguy hiểm liên tiếp",
    33: "W.205 - Đường giao nhau",
    34: "W.205e - Đường giao nhau hình chữ Y",
    35: "W.207 - Giao nhau với đường không ưu tiên",
    36: "W.208 - Giao nhau với đường ưu tiên",
    37: "W.224 - Đường người đi bộ cắt ngang",
    38: "W.225 - Trẻ em",
    39: "W.227 - Công trường",
    40: "W.244 - Đoạn hay xảy ra tai nạn",
    41: "W.245a - Đi chậm",
    42: "W.247 - Chú ý xe đỗ"
}

imgsz = (640, 640)

print(f"Model loaded successfully!")
print(f"Classes: {names}")

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

try:
    font_path = "C:/Windows/Fonts/arial.ttf"
    font = ImageFont.truetype(font_path, 18)
except:
    print("Không tìm thấy font, sử dụng font mặc định")
    font = ImageFont.load_default()

def put_vietnamese_text(img, text, position, font, color=(0, 255, 0)):
    """
    Hàm vẽ chữ tiếng Việt lên ảnh OpenCV (CHỈ DÙNG CHO CAMERA)
    """
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 180))
    
    draw.text(position, text, font=font, fill=color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def detect_image(img_path, conf_thres=0.25, iou_thres=0.45):
    im0 = cv2.imread(img_path)
    
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]
    im = im.transpose((2, 0, 1))[::-1]
    im = np.ascontiguousarray(im)
    
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()
    im /= 255
    if len(im.shape) == 3:
        im = im[None]
    
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
    
    detections = []
    annotator = Annotator(im0, line_width=3, example=str(names))
    
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            
            img_area = im0.shape[0] * im0.shape[1]
            filtered_det = []
            
            for *xyxy, conf, cls in det:
                box_width = float(xyxy[2] - xyxy[0])
                box_height = float(xyxy[3] - xyxy[1])
                box_area = box_width * box_height
                aspect_ratio = box_height / box_width if box_width > 0 else 0
                
                if (box_area / img_area >= 0.001 and 0.3 <= aspect_ratio <= 3.0):
                    filtered_det.append([*xyxy, conf, cls])
            
            for *xyxy, conf, cls in reversed(filtered_det):
                c = int(cls)
                class_name_code = names[c]  # Ký hiệu
                
                detections.append({
                    'class': vietnamese_names.get(c, "chưa xác định"),
                    'confidence': float(conf),
                    'code': class_name_code
                })
                
                label = f'{class_name_code} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
    
    im0 = annotator.result()
    return im0, detections

# ------------------ ROUTE CHÍNH ------------------
@app.route('/')
def index():
    return render_template('index.html')

def detect_video(video_path, conf_thres=0.25, iou_thres=0.45):
    """
    Xử lý video và trả về video đã detect
    """
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_path = video_path.replace('uploads', 'results').replace('.', '_result.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_detections = []
    frame_count = 0
    
    print(f"Processing video: {total_frames} frames at {fps} FPS")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # In progress mỗi 30 frames
            print(f"Processed {frame_count}/{total_frames} frames")
        
        im = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
        im = im.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(im)
        
        im_tensor = torch.from_numpy(im).to(device)
        im_tensor = im_tensor.half() if model.fp16 else im_tensor.float()
        im_tensor /= 255
        if len(im_tensor.shape) == 3:
            im_tensor = im_tensor[None]
        
        pred = model(im_tensor, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, None, False, max_det=1000)
        
        annotator = Annotator(frame, line_width=2, example=str(names))
        
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], frame.shape).round()
                
                img_area = frame.shape[0] * frame.shape[1]
                
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)
                    
                    box_width = float(xyxy[2] - xyxy[0])
                    box_height = float(xyxy[3] - xyxy[1])
                    box_area = box_width * box_height
                    aspect_ratio = box_height / box_width if box_width > 0 else 0
                    
                    if (box_area / img_area >= 0.001 and 0.3 <= aspect_ratio <= 3.0):
                        class_name_code = names[c]
                        
                        all_detections.append({
                            'frame': frame_count,
                            'class': vietnamese_names.get(c, "chưa xác định"),
                            'code': class_name_code,
                            'confidence': float(conf)
                        })
                        
                        label = f'{class_name_code} {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
        
        frame = annotator.result()
        out.write(frame)
    
    cap.release()
    out.release()
    
    print(f"Video processing completed: {frame_count} frames")
    
    unique_detections = {}
    for det in all_detections:
        key = det['code']
        if key not in unique_detections or det['confidence'] > unique_detections[key]['confidence']:
            unique_detections[key] = det
    
    return output_path, list(unique_detections.values())

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', error="Không có file được chọn")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        
        file_ext = filename.lower().split('.')[-1]
        
        if file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
            result_img, detections = detect_image(filepath, conf_thres=0.25, iou_thres=0.45)
            
            print(f"Detections found: {len(detections)}")
            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.2f}")
            print(f"{'='*60}\n")
            
            result_filename = f'result_{filename}'
            result_path = os.path.join(app.config['RESULT_FOLDER'], result_filename)
            cv2.imwrite(result_path, result_img)
            
            result_url = f'results/{result_filename}'
            
            return render_template('index.html', 
                                 result_image=result_url, 
                                 num_detections=len(detections),
                                 detections_list=detections)
        
        elif file_ext in ['mp4', 'avi', 'mov', 'mkv']:
            result_video_path, detections = detect_video(filepath, conf_thres=0.25, iou_thres=0.45)
            
            print(f"Detections found: {len(detections)}")
            for det in detections:
                print(f"  - {det['class']}: {det['confidence']:.2f}")
            print(f"{'='*60}\n")
            
            result_filename = os.path.basename(result_video_path)
            result_url = f'results/{result_filename}'
            
            return render_template('index.html', 
                                 result_video=result_url,
                                 num_detections=len(detections),
                                 detections_list=detections)
        
        else:
            return render_template('index.html', error=f"Định dạng file không được hỗ trợ: {file_ext}")
                               
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"Lỗi: {str(e)}")

# ------------------ STREAM CAMERA ------------------
def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            im = letterbox(frame, imgsz, stride=stride, auto=pt)[0]
            im = im.transpose((2, 0, 1))[::-1]
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()
            im /= 255
            if len(im.shape) == 3:
                im = im[None]
            
            pred = model(im, augment=False, visualize=False)
            pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)
            
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)
                        
                        viet_name = vietnamese_names.get(c, names[c])
                        
                        color = colors(c, True)
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        text = f"{viet_name} {conf:.2f}"
                        frame = put_vietnamese_text(frame, text, (x1, y1 - 25), font, color)
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)