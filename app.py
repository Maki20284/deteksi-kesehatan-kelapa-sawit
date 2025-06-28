import cv2
import tempfile
import streamlit as st
from ultralytics import YOLO

# Load trained model
@st.cache_resource
def load_model():
    return YOLO("yolov12_sawit.torchscript")

model = load_model()

st.title("🚀 Deteksi Kesehatan Pohon Sawit dengan YOLOv12")
st.write("""
Upload gambar pohon sawit untuk mendeteksi kesehatannya.
""")

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
    results = model(tmp_path)
    result_img = results[0].plot()

    # Convert BGR ke RGB
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    st.subheader("Hasil Deteksi")
    st.image(result_img, use_column_width=True)

    # Tampilkan detail deteksi
    st.subheader("Detail Deteksi")
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        cls_name = results[0].names[cls_id]
        conf = float(box.conf[0])
        st.write(f"- **{cls_name}** (Confidence: {conf:.2f})")