import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = "mobilenet_final.h5"  # 👈 change if needed

# Label mapping (edit if your labels differ)
label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

IMG_SIZE = (224, 224)

# -------------------------------
# LOAD MODEL
# -------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = ImageOps.fit(image, IMG_SIZE, Image.BICUBIC)
    arr = np.asarray(image).astype("float32") / 255.0
    return arr

# -----------------------`--------
# STREAMLIT UI
# -------------------------------
st.title("🩺 Diabetic Retinopathy Detection using MobileNet")
st.write("Upload a retinal image to predict the DR stage.")

uploaded_file = st.file_uploader("Upload an image (PNG or JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing image..."):
        img_arr = preprocess_image(image)
        probs = model.predict(np.expand_dims(img_arr, axis=0))[0]
        pred_idx = int(np.argmax(probs))
        pred_label = label_map.get(pred_idx, "Unknown")

    # Display results
    st.subheader("🧩 Prediction Results")
    st.write(f"**Predicted Class:** {pred_label}")
    st.write("**Class Probabilities:**")

    # Show probabilities in table
    prob_table = {label_map[i]: float(probs[i]) for i in range(len(probs))}
    st.dataframe(prob_table)

    # Optional: show probability bar chart
    st.bar_chart(prob_table)

    # Display prediction label on image
    st.image(image, caption=f"Predicted: {pred_label}", use_container_width=True)
else:
    st.info("👆 Please upload an image to get started.")
