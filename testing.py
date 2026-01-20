import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import os

# -------------------------------
# CONFIGURATION
# -------------------------------
MODEL_PATH = r"mobilenet_final.h5"  # 👈 change to your model path
IMAGE_PATH = r"diabetic_retinography_dataset\colored_images\Moderate\000c1434d8d7.png"  # 👈 change to your test image path

# Label mapping (edit if your labels differ)
label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferate_DR"
}

# -------------------------------
# LOAD MODEL
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found at {IMAGE_PATH}")

print("✅ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (224, 224)

# -------------------------------
# PREPROCESS IMAGE
# -------------------------------
def load_and_preprocess(path):
    img = Image.open(path).convert("RGB")
    img = ImageOps.fit(img, IMG_SIZE, Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0  # normalize 0–1
    return arr

print("🖼️  Loading and preprocessing image...")
img_arr = load_and_preprocess(IMAGE_PATH)

# -------------------------------
# PREDICT
# -------------------------------
probs = model.predict(np.expand_dims(img_arr, axis=0))[0]
pred_idx = int(np.argmax(probs))
pred_label = label_map.get(pred_idx, "Unknown")

print("\n===============================")
print(f"Predicted label index: {pred_idx}")
print(f"Predicted class: {pred_label}")
print("===============================\n")

# Show class probabilities
print("Class probabilities:")
for i in np.argsort(probs)[::-1]:
    print(f"  {label_map.get(i,'?')}: {probs[i]:.4f}")

# -------------------------------
# DISPLAY IMAGE (optional)
# -------------------------------
plt.imshow(img_arr)
plt.axis('off')
plt.title(f"Predicted: {pred_label}")
plt.show()
