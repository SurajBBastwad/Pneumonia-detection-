import os
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('medical_ai_model.h5')

# Function to preprocess the uploaded image
def load_and_preprocess_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    img = cv2.resize(img, (256, 256))  # Resize to (256, 256)
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (1 channel for grayscale)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image
    return img

# Route to display the upload form
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # Save the file to a temporary location
    file_path = os.path.join('uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # Preprocess the image
    img_array = load_and_preprocess_image(file_path)

    # Predict using the trained model
    prediction = model.predict(img_array)

    # Determine the result
    result = 'PNEUMONIA' if prediction[0][0] > 0.5 else 'NORMAL'

    # Clean up the saved file after processing
    os.remove(file_path)

    # Return the result as part of the result page
    return render_template('result.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
