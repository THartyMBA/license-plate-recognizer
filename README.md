# license-plate-recognizer

🚗🔢 License Plate & Vehicle Info Extractor
A Streamlit proof-of-concept that detects a car’s license plate from a rear-view image, OCRs the plate text, and annotates the vehicle with a rough “make/model” label via ResNet50.

Demo only—no production-grade ALPR datasets, accuracy benchmarks, or compliance controls.
For enterprise computer-vision systems, contact me.

🔍 What it does
Upload a rear-view car image (.jpg, .jpeg, .png).

Detect the car bounding box using YOLOv8n on CPU.

OCR text regions via EasyOCR, filtering for license-plate-like aspect ratios.

Annotate the image with:

Blue box & “car” label around the vehicle

Red box & plate number above the license plate

Green text below the car box indicating the top-1 ImageNet class as a proxy “make/model”

Display the annotated image.

Export a CSV of detected plates (text, confidence, bounding box) and the vehicle label.

✨ Key Features
End-to-end CV pipeline: object detection → OCR → image classification

CPU-friendly: uses YOLOv8n (27 MB), EasyOCR, and ResNet50 with no GPU required

Single-file app: all logic in license_plate_recognizer.py

Interactive: upload, view annotations, and download results in one place

Zero secrets: no API keys needed

🚀 Quick Start (Local)
bash
Copy
Edit
git clone https://github.com/THartyMBA/license-plate-recognizer.git
cd license-plate-recognizer
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run license_plate_recognizer.py
Open http://localhost:8501.

Upload your rear-view car photo.

View the annotated image and download the results.csv.

☁️ Deploy on Streamlit Cloud
Push this repo (public or private) under THartyMBA to GitHub.

Visit streamlit.io/cloud → New app → select your repo & branch → Deploy.

Share your live URL—no further configuration needed.

🛠️ Requirements
shell
Copy
Edit
streamlit>=1.32
ultralytics
easyocr
torch
torchvision
Pillow
numpy
requests
(All CPU-compatible wheels—suitable for Streamlit’s free tier.)

🗂️ Repo Structure
kotlin
Copy
Edit
license-plate-recognizer/
├─ license_plate_recognizer.py   ← single-file Streamlit app  
├─ requirements.txt  
└─ README.md                     ← you’re reading this  
📜 License
CC0 1.0 – public-domain dedication. Attribution appreciated but not required.

🙏 Acknowledgements
Streamlit – rapid Python UIs

Ultralytics YOLO – object-detection framework

EasyOCR – text detection & recognition

PyTorch & torchvision – ResNet50 classification

Pillow & NumPy – image processing

Detect plates and label vehicles in seconds—enjoy! 🎉
