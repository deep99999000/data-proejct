import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained Keras model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mymodel.keras")
    return model

model = load_model()

# Define class names based on your training labels
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Streamlit UI
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Classification App")
st.write("Upload an image of garbage to predict its category.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((224, 224))  # Match model input shape
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    # Display result
    st.success(f"Prediction: **{predicted_class.capitalize()}**")
    st.info(f"Confidence: {confidence * 100:.2f}%")

