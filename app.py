import streamlit as st
import tempfile
from ultralytics import YOLO
import cv2
from PIL import Image
from streamlit_cropper import st_cropper
import numpy as np
import pandas as pd

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("yolov12_sawit_v2.torchscript")

model = load_model()

st.title("🚀 Deteksi Kesehatan Pohon Sawit dengan YOLOv12")
st.write("""
Upload gambar pohon sawit untuk mendeteksi kesehatannya.
""")

# Sidebar Settings
ALL_CLASSES = ['Dead', 'Grass', 'Healthy', 'Small', 'Yellow']
st.sidebar.header("⚙️ Pengaturan Deteksi")
selected_classes = st.sidebar.multiselect(
    'Filter kelas yang ingin ditampilkan:',
    ALL_CLASSES,
    default=ALL_CLASSES
)
conf_threshold = st.sidebar.slider('Confidence Threshold', 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider('IoU Threshold', 0.1, 1.0, 0.4, 0.05)
aspect_ratio_setting = st.sidebar.selectbox(
    "Aspect Ratio untuk Crop",
    options=["Bebas", "1:1"],
    index=0
)
if aspect_ratio_setting == "1:1":
    aspect_ratio = (1, 1)
else:
    aspect_ratio = None

# Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.subheader("📷 Gambar Asli")
    img = Image.open(uploaded_file)
    img = img.convert("RGB")
    st.image(img, use_column_width=True)

    st.subheader("🔍 Pilih Area untuk Crop (Zoom & Pan)")
    cropped_img = st_cropper(
        img,
        aspect_ratio=aspect_ratio,
        box_color='blue',
        realtime_update=True
    )

    if cropped_img is not None:
        st.subheader("🖼️ Hasil Crop")
        st.image(cropped_img, use_column_width=True, caption="Bagian gambar yang dipilih")

        # ✅ Simpan hasil crop ke file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_crop:
            cropped_img.save(tmp_crop, format='JPEG')
            tmp_crop_path = tmp_crop.name

        # ✅ Inferensi model hanya pada hasil crop
        results = model(tmp_crop_path, conf=conf_threshold)

        boxes = results[0].boxes
        names = results[0].names

        if len(boxes) == 0:
            st.warning("Tidak terdeteksi objek pada gambar dengan threshold ini. Coba turunkan threshold confidence atau coba perbesar gambarnya.")
        else:
            # Load gambar hasil crop
            img_cv = cv2.imread(tmp_crop_path)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            img_filtered = img_cv.copy()

            # Warna untuk tiap kelas
            COLOR_MAP = {
                'Dead': (255, 0, 0),
                'Grass': (0, 255, 0),
                'Healthy': (0, 200, 0),
                'Small': (255, 255, 0),
                'Yellow': (255, 165, 0)
            }

            # Font settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 4

            detections = []
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = names[cls_id]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if conf >= conf_threshold and cls_name in selected_classes:
                    color = COLOR_MAP.get(cls_name, (255, 255, 255))
                    # Kotak
                    cv2.rectangle(img_filtered, (x1, y1), (x2, y2), color, 2)
                    # Label
                    label_text = f"{cls_name} {conf:.2f}"
                    (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                    cv2.rectangle(img_filtered, (x1, y1 - text_h - baseline), (x1 + text_w, y1), color, -1)
                    cv2.putText(img_filtered, label_text, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness)

                    detections.append({
                        "Label": cls_name,
                        "Confidence": f"{conf:.2f}",
                        "X1": x1,
                        "Y1": y1,
                        "X2": x2,
                        "Y2": y2
                    })

            st.subheader("✅ Hasil Deteksi (Filtered)")
            st.image(img_filtered, use_column_width=True, caption="Bounding box hanya untuk kelas yang dipilih")

            if detections:
                df = pd.DataFrame(detections)
                st.subheader("📋 Tabel Deteksi")
                st.dataframe(df, use_container_width=True)
            else:
                st.info("Tidak ada deteksi yang melewati filter kelas atau threshold.")
