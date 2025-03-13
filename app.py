from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
import io
import base64
import os
import time
import random  # For demo purposes

app = Flask(__name__)

# Path to the face detection model
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    if request.method == 'POST':
        try:
            # Get the image data from the POST request
            image_data = request.json.get('image')
            if not image_data:
                return jsonify({"error": "No image data received"}), 400
            
            # Remove the data:image/jpeg;base64, prefix
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                return jsonify({"error": "Could not decode image"}), 400
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # If no faces are found
            if len(faces) == 0:
                return jsonify({
                    "error": "No faces detected in the image. Please try another image with a clear face."
                }), 400
            
            # For simplicity, just use the first face detected
            (x, y, w, h) = faces[0]
            
            # For demo purposes, determine "race" based on simple image features
            # Note: This is NOT a real race detection algorithm, just for demo
            face_img = img[y:y+h, x:x+w]
            
            # Extract image features (for demonstration only)
            avg_color = np.mean(face_img, axis=(0, 1))
            brightness = np.mean(cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY))
            
            # In a real app, you'd use a proper ML model here
            # This is just a placeholder logic for demonstration
            race_categories = ["White", "Black", "Asian", "Indian", "Other"]
            
            # Deterministic random for demo consistency (based on image features)
            random.seed(int(sum(avg_color) + brightness))
            
            # Generate dummy probabilities (FOR DEMO ONLY)
            probabilities = [random.random() for _ in range(len(race_categories))]
            total = sum(probabilities)
            normalized_probs = [p/total for p in probabilities]
            
            race_data = dict(zip(race_categories, normalized_probs))
            dominant_race = max(race_data.items(), key=lambda x: x[1])
            
            return jsonify({
                "race": dominant_race[0],
                "confidence": dominant_race[1] * 100,
                "all_races": race_data,
                "message": f"Detected race: {dominant_race[0]} with {dominant_race[1]:.2%} confidence",
                "note": "This is a demonstration only and not a real race detector."
            })
                
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "Method not allowed"}), 405

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)