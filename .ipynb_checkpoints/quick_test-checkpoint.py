{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43332432-be94-47e5-a03d-0d903181cee9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import cv2\n",
    "\n",
    "# Load the trained model\n",
    "model = tf.keras.models.load_model(\"medical_ai_model.h5\")\n",
    "\n",
    "# Load and process the image\n",
    "image_path = \"C:/Users/Suraj/Desktop/OIP.jpg\"  # Path to your image\n",
    "image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)\n",
    "\n",
    "if image is None:\n",
    "    print(\"❌ Failed to load image.\")\n",
    "else:\n",
    "    image = cv2.resize(image, (256, 256))\n",
    "    image = image / 255.0\n",
    "    image = np.expand_dims(image, axis=-1)  # Add channel dimension\n",
    "    image = np.expand_dims(image, axis=0)   # Add batch dimension\n",
    "\n",
    "    # Predict\n",
    "    prediction = model.predict(image)\n",
    "    confidence = prediction[0][0]\n",
    "    result = \"🧬 Pneumonia Detected\" if confidence > 0.5 else \"✅ Normal\"\n",
    "    print(f\"🔍 Prediction Result: {result}\")\n",
    "    print(f\"📊 Confidence Score: {confidence*100:.2f}%\")\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
