import streamlit as st
import numpy as np
<<<<<<< HEAD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load model
model = load_model("mask_model.h5")

st.title("😷 Face Mask Detection App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(100, 100))
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        st.error("❌ No Mask Detected")
    else:
        st.success("✅ Mask Detected")
=======
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
>>>>>>> 007492d267a677ed8b892658af2bb407f42c3541
