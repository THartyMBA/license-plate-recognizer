# license_plate_recognizer.py
"""
License Plate & Vehicle Info Extractor  🚗🔢
────────────────────────────────────────────────────────────────────────────
Upload a photo of a car’s rear. This POC:
1. Detects the car bounding box via YOLOv8n.
2. Uses EasyOCR to find and OCR text regions; heuristically filters for license plates.
3. Annotates the image with the plate number and car bounding box.
4. Classifies the vehicle (ImageNet top-1) within the car box as a rough “make/model” proxy.
5. Displays results and lets you download a CSV with plate text and vehicle label.

*Demo only*—no production-grade accuracy or datasets.
For enterprise CV and ALPR pipelines, [contact me](https://drtomharty.com/bio).
"""

import io
import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import easyocr
from ultralytics import YOLO
import torch
from torchvision import models, transforms

# ───────────────────────────────────────── Model loaders ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_yolo():
    return YOLO("yolov8n.pt")  # detects “car” class

@st.cache_resource(show_spinner=False)
def load_ocr():
    return easyocr.Reader(["en"], gpu=torch.cuda.is_available())

@st.cache_resource(show_spinner=False)
def load_classifier():
    model = models.resnet50(pretrained=True)
    model.eval()
    return model

@st.cache_resource(show_spinner=False)
def load_imagenet_labels():
    import requests
    resp = requests.get(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    )
    return resp.text.splitlines()

# preprocessing transform for classifier
clf_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ─────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="License Plate & Vehicle Info", layout="wide")
st.title("🚗 License Plate & Vehicle Info Extractor")

st.info(
    "🔔 **Demo Notice**  \n"
    "This is a proof-of-concept using YOLOv8, EasyOCR, and ResNet50.  \n"
    "For production ALPR or vehicle-recognition systems, [contact me](https://drtomharty.com/bio).",
    icon="💡"
)

uploaded = st.file_uploader("Upload rear-view car image", type=["jpg","jpeg","png"])
if not uploaded:
    st.stop()

# load models
yolo = load_yolo()
ocr  = load_ocr()
clf  = load_classifier()
labels = load_imagenet_labels()

# process image
img = Image.open(uploaded).convert("RGB")
arr = np.array(img)

# 1. Detect cars
results = yolo(arr)
car_boxes = []
for box, cls in zip(results[0].boxes.xyxy, results[0].boxes.cls):
    if yolo.names[int(cls)] == "car":
        x1, y1, x2, y2 = map(int, box)
        car_boxes.append((x1, y1, x2, y2))

draw = ImageDraw.Draw(img)

# annotate cars
for (x1,y1,x2,y2) in car_boxes:
    draw.rectangle([x1,y1,x2,y2], outline="blue", width=3)
    draw.text((x1, y1-12), "car", fill="blue")

# 2. OCR text regions
ocr_results = ocr.readtext(arr)

# heuristic: width/height > 3 and region near bottom half of image
h_img, w_img, _ = arr.shape
plates = []
for bbox, text, conf in ocr_results:
    # bbox: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    xs = [int(p[0]) for p in bbox]
    ys = [int(p[1]) for p in bbox]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    bw, bh = x2-x1, y2-y1
    if bw > 3*bh and conf > 0.3 and y1 > h_img//3:
        plates.append((x1,y1,x2,y2,text,conf))
# annotate plates
for x1,y1,x2,y2,text,conf in plates:
    draw.rectangle([x1,y1,x2,y2], outline="red", width=3)
    draw.text((x1, y1-12), text, fill="red")

# 3. Classify vehicle make/model (ImageNet proxy) on first car
vehicle_label = "N/A"
if car_boxes:
    x1,y1,x2,y2 = car_boxes[0]
    car_crop = img.crop((x1,y1,x2,y2))
    inp = clf_transform(car_crop).unsqueeze(0)
    with torch.no_grad():
        out = clf(inp)
        idx = int(out[0].argmax())
        vehicle_label = labels[idx]
    draw.text((x1, y2+4), vehicle_label, fill="green")

# 4. Display annotated image
st.subheader("Annotated Image")
st.image(img, use_column_width=True)

# 5. Summarize and download results
rows = []
for i, (x1,y1,x2,y2,text,conf) in enumerate(plates, start=1):
    rows.append({
        "plate_index": i,
        "text": text,
        "confidence": conf,
        "bbox": f"{x1},{y1},{x2},{y2}"
    })
if not rows:
    st.warning("No license plate detected. Try a clearer image.")
rows.append({"plate_index": "vehicle_label", "text": vehicle_label, "confidence": "", "bbox": ""})
df = st.dataframe(rows)

csv = "\n".join([",".join(map(str,row.values())) for row in rows])
st.download_button("⬇️ Download results CSV", csv.encode(), "results.csv", "text/csv")
