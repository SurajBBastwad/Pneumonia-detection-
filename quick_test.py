import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("medical_ai_model.h5")

# Load and process the image
image_path = "C:/Users/Suraj/Desktop/OIP.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("❌ Failed to load image.")
else:
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = np.expand_dims(image, axis=0)   # Add batch dimension

    prediction = model.predict(image)
    confidence = prediction[0][0]
    result = "🧬 Pneumonia Detected" if confidence > 0.5 else "✅ Normal"
    print(f"🔍 Prediction Result: {result}")
    print(f"📊 Confidence Score: {confidence*100:.2f}%")
