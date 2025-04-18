from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model (make sure the path is correct)
model = tf.keras.models.load_model("medical_ai_model.h5")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    
    # Read and preprocess the image
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256)) / 255.0
    image = image.reshape(1, 256, 256, 1)

    # Make prediction
    prediction = model.predict(image)[0][0]
    result = "Pneumonia Detected" if prediction > 0.5 else "Normal"
    return jsonify({"Prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
