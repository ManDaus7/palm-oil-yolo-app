import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Title
st.title("🌴 Palm Oil Fruit Ripeness Detection")

# Load model
model = YOLO("best.pt")

# Upload Image
uploaded_file = st.file_uploader("Upload a palm fruit bunch image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_container_width=True)

    st.write("Detecting...")

    # Convert PIL to numpy
    image_np = np.array(image)

    # Predict using YOLOv8
    results = model.predict(source=image_np, conf=0.5, imgsz=640)

    # Show result
    for r in results:
        annotated_frame = r.plot()  # returns numpy image with boxes and labels
        st.image(annotated_frame, caption="Detection Result", use_container_width=True)

