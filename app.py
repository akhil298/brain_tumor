from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the trained model
model = load_model('Brain Tumor.h5')

# Define a function to preprocess the image
def preprocess_image(image_bytes):
    # Read the image file as RGB
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    # Resize the image to match the input size of the model
    image = image.resize((224, 224))
    # Convert image to array
    image_array = np.array(image)
    # Normalize and scale pixel values to be in the range [0, 1]
    image_array = image_array / 255.0
    # Add batch dimension to match the expected input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to predict
def predict(image_bytes):
    # Preprocess the image
    processed_image = preprocess_image(image_bytes)
    # Make prediction
    prediction = model.predict(processed_image)
    # Decode the prediction
    if prediction[0][0] > 0.5:
        result = "Tumor present"
    else:
        result = "No tumor"
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image_bytes = file.read()
        prediction_result = predict(image_bytes)
        return jsonify({'result': prediction_result})

if __name__ == '__main__':
    app.run(debug=True)
