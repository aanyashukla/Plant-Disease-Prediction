import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/plant_disease_model_v1.h5"
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using Pillow
def preprocess_image(img, target_size=(224, 224)):
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


#Prediction
def predict(img):
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0]
    top_idx = np.argmax(prediction)
    confidence = prediction[top_idx]
    return top_idx, confidence, prediction

# Streamlit App
st.title('Plant Disease Classifier')

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.markdown("🔍 **Predicting...**")
    top_idx, confidence, all_preds = predict(image)
    predicted_class = class_indices[str(top_idx)]

    st.success(f"✅ **Predicted Disease:** `{predicted_class}`")
    st.info(f"📈 Confidence: `{confidence*100:.2f}%`")

    # Top-3 predictions
    top_3_indices = np.argsort(all_preds)[-3:][::-1]
    st.markdown("📊 **Top 3 Predictions:**")
    for i in top_3_indices:
        st.write(f"{class_indices[str(i)]}: `{all_preds[i]*100:.2f}%`")

    st.toast("Prediction complete!", icon="✅")
