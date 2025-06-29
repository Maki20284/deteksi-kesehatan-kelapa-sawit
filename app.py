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

st.sidebar.header("⚙️ Pengaturan Deteksi")
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
        # Ambil image hasil plot dengan threshold
        result_img = results[0].plot()
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        st.subheader("Hasil Deteksi")
        st.image(result_img, use_column_width=True, caption="Deteksi dengan bounding box dan label")


    # Buat dataframe hasil deteksi
        detections = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if conf >= conf_threshold:
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