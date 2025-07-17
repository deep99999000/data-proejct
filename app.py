import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mymodel.keras")

model = load_model()

# Class labels - adjust if different
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Streamlit UI
st.set_page_config(page_title="Garbage Classifier", layout="centered")

st.title("🗑️ Garbage Waste Classifier")
st.markdown("Upload an image to predict the garbage category.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"**Prediction:** {predicted_class.capitalize()} ({confidence:.2f}% confidence)")
