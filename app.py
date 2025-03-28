import os
from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
IMG_SIZE = 224  # Must match your model's expected input size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='dog_breed_classifier.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load class names
with open('class_names.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and normalize with explicit float32 conversion
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]
    return np.expand_dims(image_array, axis=0)  # Add batch dimension

def predict_single_image(image_array):
    """Predict breed probabilities for a single image"""
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]

@app.route('/')
def home():
    return 'Dog Breed Offspring Prediction API - Send POST request with two dog images to /predict-offspring'

@app.route('/predict-offspring', methods=['POST'])
def predict_offspring():
    """Handle offspring prediction from two parent images"""
    try:
        # Check if both images are present
        if 'image1' not in request.files or 'image2' not in request.files:
            return jsonify({'error': 'Both image1 and image2 files are required'}), 400
        
        image1 = request.files['image1']
        image2 = request.files['image2']
        
        # Validate file types
        if not (allowed_file(image1.filename) and allowed_file(image2.filename)):
            return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400
        
        # Process images
        try:
            img1 = Image.open(io.BytesIO(image1.read()))
            img2 = Image.open(io.BytesIO(image2.read()))
            
            img1_processed = preprocess_image(img1)
            img2_processed = preprocess_image(img2)
        except Exception as e:
            return jsonify({'error': f'Image processing failed: {str(e)}'}), 400
        
        # Verify input shapes match model expectations
        expected_shape = input_details[0]['shape']
        if img1_processed.shape != tuple(expected_shape) or img2_processed.shape != tuple(expected_shape):
            return jsonify({
                'error': f'Input shape mismatch. Expected {expected_shape}, got {img1_processed.shape}'
            }), 400
        
        # Run inference for both images
        try:
            # Get predictions for both parents
            pred1 = predict_single_image(img1_processed)
            pred2 = predict_single_image(img2_processed)
            
            # Combine probabilities (average)
            combined_probs = (pred1 + pred2) / 2
        except Exception as e:
            return jsonify({'error': f'Model inference failed: {str(e)}'}), 500
        
        # Process combined predictions
        top_k = 3
        top_indices = np.argsort(combined_probs)[-top_k:][::-1]
        
        # Get parent predictions
        parent1_top_idx = np.argmax(pred1)
        parent2_top_idx = np.argmax(pred2)
        
        results = {
            'parent1': {
                'breed': class_names[parent1_top_idx],
                'confidence': float(pred1[parent1_top_idx])
            },
            'parent2': {
                'breed': class_names[parent2_top_idx],
                'confidence': float(pred2[parent2_top_idx])
            },
            'offspring_predictions': [
                {
                    'breed': class_names[i],
                    'confidence': float(combined_probs[i])
                } for i in top_indices
            ],
            'top_offspring_prediction': {
                'breed': class_names[top_indices[0]],
                'confidence': float(combined_probs[top_indices[0]])
            }
        }
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, threaded=True)