from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import os
import random

# Set random seeds for consistent predictions
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

app = Flask(__name__)

# Load the model once
model = from_pretrained_keras('Emmawang/mobilenet_v2_fake_image_detection')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    if request.method == 'POST':
        if 'image' not in request.files:
            result = "No file uploaded."
        else:
            file = request.files['image']
            if file.filename == '':
                result = "No selected file."
            else:
                try:
                    # Prepare image
                    img = Image.open(file).resize((128, 128)).convert('RGB')
                    img = np.array(img).astype(np.float32) / 255.0
                    img = img.reshape(-1, 128, 128, 3)

                    # Predict
                    prediction = model.predict(img)[0]  # Single prediction
                    label = np.argmax(prediction)
                    confidence = float(prediction[label]) * 100  # Convert to %

                    # Logic based on confidence
                    if label == 0 and confidence >= 70:
                        result = f"Real Image (Confidence: {confidence:.2f}%)"
                    elif label == 1:
                        result = f"Fake Image (Confidence: {confidence:.2f}%)"
                    else:
                        result = f"Uncertain Real Image (Confidence: {confidence:.2f}%)"
                except Exception as e:
                    result = f"Error processing image: {str(e)}"

    return render_template('index.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)
