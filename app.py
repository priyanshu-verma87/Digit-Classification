import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load your trained model
model = load_model("digit_model.h5")

# Title
st.title("🧠 Handwritten Digit Classifier")
st.write("Upload an image of a digit (0–9), and I will predict it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img_np = np.array(image)

    # Resize and preprocess
    img = cv2.resize(img_np, (28, 28))
    img = cv2.bitwise_not(img)  # Invert if needed
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    # Show uploaded image
    st.image(image, caption="Uploaded Digit", use_container_width=True)

    # Predict
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)

    st.success(f"✅ Predicted Digit: **{predicted_class}**")
