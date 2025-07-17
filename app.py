import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mymodel.keras")

model = load_model()

# Class labels
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# App layout
st.set_page_config(page_title="Garbage Classifier", layout="centered")
st.title("🗑️ Garbage Classification App")
st.write("Upload an image to classify it into one of the 6 garbage categories.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((150, 150))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}%)")
