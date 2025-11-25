import torch
import pathlib
from pathlib import Path
from models.common import DetectMultiBackend  # từ YOLOv5 repo
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
import cv2

# Patch PosixPath
pathlib.PosixPath = str

# Load model
weights = "best.pt"
device = select_device('cpu')  # hoặc '0' nếu có GPU
model = DetectMultiBackend(weights, device=device)

# Load ảnh
img_path = "camdoxe.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize, chuẩn bị tensor
from utils.datasets import letterbox
import torch

img_resized = letterbox(img_rgb, new_shape=640)[0]
img_resized = img_resized.transpose((2, 0, 1))  # HWC to CHW
img_resized = torch.from_numpy(img_resized).float() / 255.0
img_resized = img_resized.unsqueeze(0)  # batch dimension

# Inference
pred = model(img_resized)
pred = non_max_suppression(pred)

# Vẽ bounding box
from utils.plots import Annotator, colors

annotator = Annotator(img)
for det in pred:  # det: [x1, y1, x2, y2, conf, cls]
    if det is not None and len(det):
        for *xyxy, conf, cls in det:
            c = int(cls)
            label = f"{c} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(c, True))

result_img = annotator.result()
cv2.imwrite("result.jpg", result_img)
print("Saved result.jpg")
