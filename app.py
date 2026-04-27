import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("mask_model.h5")

st.title("Face Mask Detection")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_resized = cv2.resize(img, (224, 224))
    img_resized = img_resized / 255.0
    img_resized = np.reshape(img_resized, (1, 224, 224, 3))

    pred = model.predict(img_resized)

    if pred[0][0] > 0.5:
        st.error("No Mask 😷❌")
    else:
        st.success("Mask Detected 😷✅")

    st.image(img, channels="BGR")