import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO
import pandas as pd
import tempfile
import os

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("yolov12_sawit_v2.torchscript")

model = load_model()

st.title("🚀 Deteksi Kesehatan Pohon Sawit dengan YOLOv12")
st.write("""
Upload gambar pohon sawit untuk mendeteksi kesehatannya.
""")

# Semua nama kelas yang tersedia (static list atau dari model)
ALL_CLASSES = ['Dead', 'Grass', 'Healthy', 'Small', 'Yellow']

# Sidebar filter kelas
st.sidebar.header("⚙️ Pengaturan Deteksi")
selected_classes = st.sidebar.multiselect(
    'Filter kelas yang ingin ditampilkan:',
    ALL_CLASSES,
    default=ALL_CLASSES
)
conf_threshold = st.sidebar.slider('Confidence Threshold', 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider('IoU Threshold', 0.1, 1.0, 0.4, 0.05)

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.subheader("Gambar Asli")
    st.image(uploaded_file, use_column_width=True)

    # Simpan gambar ke file sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Jalankan prediksi YOLO
    results = model(tmp_path, conf=conf_threshold)

    boxes = results[0].boxes
    names = results[0].names
    
    if len(boxes) == 0:
        st.warning("Tidak terdeteksi objek pada gambar dengan threshold ini. Coba turunkan threshold confidence.")
    else:
        # Baca gambar asli
        original_img = cv2.imread(tmp_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        img_filtered = original_img.copy()

        # Warna untuk tiap kelas (opsional)
        COLOR_MAP = {
            'Dead': (255, 0, 0),
            'Grass': (0, 255, 0),
            'Healthy': (0, 200, 0),
            'Small': (255, 255, 0),
            'Yellow': (255, 165, 0)
        }

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf >= conf_threshold and cls_name in selected_classes:
                color = COLOR_MAP.get(cls_name, (255, 255, 255))
                # Gambar kotak
                cv2.rectangle(img_filtered, (x1, y1), (x2, y2), color, 2)
                # Label teks
                label_text = f"{cls_name} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
                cv2.rectangle(img_filtered, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
                cv2.putText(img_filtered, label_text, (x1, y1 - baseline), font, font_scale, (0, 0, 0), thickness)

        st.subheader("Hasil Deteksi (Filtered)")
        st.image(img_filtered, use_column_width=True, caption="Bounding box hanya untuk kelas yang dipilih")

    # Buat dataframe hasil deteksi
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf >= conf_threshold and cls_name in selected_classes:
                detections.append({
                    "Label": cls_name,
                    "Confidence": f"{conf:.2f}",
                    "X1": x1,
                    "Y1": y1,
                    "X2": x2,
                    "Y2": y2
                })

        if detections:
            df = pd.DataFrame(detections)
            st.subheader("📋 Tabel Deteksi")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Tidak ada deteksi yang melewati threshold.")