import streamlit as st
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# 🔽 DOWNLOAD MODEL FROM GOOGLE DRIVE
# -------------------------------

file_id = "PASTE_YOUR_REAL_FILE_ID_HERE"   # 🔴 IMPORTANT
output = "mask_model.h5"

if not os.path.exists(output):
    try:
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, output, quiet=False)
    except:
        st.error("❌ Model download failed")
        st.info("👉 Check: file is public & file_id is correct")
        st.stop()

# -------------------------------
# 🤖 DUMMY PREDICTION
# -------------------------------
def predict(img):
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