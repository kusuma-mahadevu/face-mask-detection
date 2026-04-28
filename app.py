import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# 📥 Download model from Google Drive
# -------------------------------
file_id = "PASTE_YOUR_FILE_ID_HERE"   # 🔴 replace this
output = "mask_model.h5"

if not os.path.exists(output):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        gdown.download(url, output, quiet=False)
    except Exception as e:
        st.error("❌ Model download failed. Check Google Drive link.")
        st.stop()

# -------------------------------
# 🤖 Dummy Prediction (since no TensorFlow)
# -------------------------------
def predict(img):
    # Random prediction (demo purpose)
    return np.random.choice(["Mask", "No Mask"])

# -------------------------------
# 🎨 UI
# -------------------------------
st.title("😷 Face Mask Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    result = predict(img)

    if result == "Mask":
        st.success("✅ Mask Detected")
    else:
        st.error("❌ No Mask Detected")