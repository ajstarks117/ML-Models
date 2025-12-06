import io
import os
import uuid
import datetime
import base64
import traceback
from flask import Flask, request, jsonify, render_template
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# --- CONFIGURATION ---
DETECT_MODEL_PATH = "runs/detect/yolo_tree_mixed3/weights/best.pt"
CLASSIFY_MODEL_PATH = "runs/classify/my_species_model3/weights/best.pt"

# Feedback Folders
FEEDBACK_DIR = "feedback_data"
os.makedirs(f"{FEEDBACK_DIR}/aerial", exist_ok=True)
os.makedirs(f"{FEEDBACK_DIR}/species", exist_ok=True)

print("⏳ Loading models... (Do not close this window)")
try:
    detect_model = YOLO(DETECT_MODEL_PATH)
    classify_model = YOLO(CLASSIFY_MODEL_PATH)
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR: Could not load models.\n{e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        
        # Get confidence from slider (default to 25%)
        conf_val = float(request.form.get('confidence', 25)) / 100.0

        img = Image.open(file.stream).convert('RGB')
        
        # Run Inference with dynamic confidence
        results = detect_model.predict(img, conf=conf_val)
        r = results[0]
        count = len(r.boxes)
        
        res_plotted = r.plot()
        _, buffer = cv2.imencode('.jpg', res_plotted)
        img_str = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'count': count, 'image_data': img_str})

    except Exception as e:
        print("\n❌ ERROR IN /DETECT:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file part'}), 400
        file = request.files['file']

        img = Image.open(file.stream).convert('RGB')
        
        results = classify_model.predict(img)
        r = results[0]
        
        top_prob = r.probs.top1
        conf = r.probs.top1conf.item()
        species_name = r.names[top_prob].upper()

        # Only return image if confidence is high enough (optional check)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        img_str = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({'species': species_name, 'confidence': round(conf * 100, 2), 'image_data': img_str})

    except Exception as e:
        print("\n❌ ERROR IN /CLASSIFY:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
# --- NEW FEEDBACK ROUTE ---
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
        file = request.files['file']
        mode = request.form.get('mode', 'unknown') # 'aerial' or 'species'
        
        # Generate unique filename: timestamp_uuid.jpg
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.jpg"
        
        # Save to specific folder
        save_path = os.path.join(FEEDBACK_DIR, mode, filename)
        
        # Save file
        file.save(save_path)
        
        print(f"📝 Feedback received! Saved incorrect image to: {save_path}")
        return jsonify({'message': 'Image saved for retraining', 'path': save_path})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)