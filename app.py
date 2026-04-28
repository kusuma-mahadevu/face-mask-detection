import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# Download model from Google Drive
file_id = "YOUR_FILE_ID"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists("mask_model.h5"):
    gdown.download(url, "mask_model.h5", quiet=False)

# Dummy prediction (since no tensorflow)
def predict(img):
    return np.random.choice(["Mask", "No Mask"])

st.title("😷 Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img)

    result = predict(img)

    if result == "Mask":
        st.success("✅ Mask Detected")
    else:
        st.error("❌ No Mask Detected")