import io
import os
from flask import Flask, request, render_template_string, send_file
from PIL import Image
from ultralytics import YOLO

# HTML Template
HTML = """
<!doctype html>
<title>Tree Species Identifier</title>
<style>
  body { font-family: sans-serif; text-align: center; padding: 50px; }
  .result { margin-top: 20px; font-size: 24px; font-weight: bold; color: #27ae60; }
  img { margin-top: 20px; border-radius: 10px; max-width: 400px; }
</style>

<h2>🧬 Tree Species Identifier</h2>
<p>Upload a photo of a tree to identify its species.</p>

<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*" required>
  <br><br>
  <input type=submit value="Identify Species 🔍">
</form>

{% if result %}
  <div class="result">Result: {{ result.name }} ({{ result.conf }}%)</div>
  <img src="/uploaded_image" />
{% endif %}
"""

app = Flask(__name__)
_app_state = {"img_bytes": None}

# LOAD YOUR TRAINED MODEL HERE
# Check 'runs/classify/my_species_model/weights/best.pt' after training
MODEL_PATH = r"runs/classify/my_species_model3/weights/best.pt"

print("Loading Species Model...")
try:
    model = YOLO(MODEL_PATH)
except:
    print("Warning: Model not found. Run train_species_cls.py first!")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        if 'file' not in request.files: return render_template_string(HTML)
        file = request.files['file']
        if file.filename == '': return render_template_string(HTML)

        # Read image
        img = Image.open(file.stream).convert('RGB')
        
        # Save for display
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        buf.seek(0)
        _app_state["img_bytes"] = buf.read()

        # INFERENCE
        if model:
            # Predict the class
            results = model(img) 
            # Get the top prediction
            top_prob = results[0].probs.top1
            top_conf = results[0].probs.top1conf.item()
            class_name = results[0].names[top_prob]
            
            result = {
                "name": class_name.upper(), 
                "conf": round(top_conf * 100, 1)
            }

    return render_template_string(HTML, result=result)

@app.route('/uploaded_image')
def uploaded_image():
    if _app_state["img_bytes"] is None: return "No image", 404
    return send_file(io.BytesIO(_app_state["img_bytes"]), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(port=5001, debug=True) # Running on port 5001 to not conflict with your other app