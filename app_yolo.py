import io
import os
from flask import Flask, request, render_template_string, send_file
from PIL import Image
from ultralytics import YOLO
import numpy as np

# HTML Template for the UI
HTML = """
<!doctype html>
<title>TreeTrackAI - YOLOv8</title>
<style>
  body { font-family: sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; text-align: center; }
  form { margin: 20px 0; padding: 20px; border: 1px solid #ccc; border-radius: 10px; background: #f9f9f9; }
  img { margin-top: 20px; border: 2px solid #333; border-radius: 5px; max-width: 100%; height: auto; }
  .stats { font-size: 1.2em; color: #2c3e50; font-weight: bold; }
</style>

<h2>🌲 TreeTrackAI: YOLO Detection</h2>
<p>Upload an aerial image to count trees using your trained YOLOv8 model.</p>

<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*" required>
  <br><br>
  <label>Confidence Threshold (0-1): 
    <input type=number name=threshold step=0.05 min=0.05 max=1 value="{{ result.threshold if result else default_threshold }}">
  </label>
  <br><br>
  <input type=submit value="Detect Trees 🚀" style="cursor: pointer; padding: 10px 20px; font-size: 1em;">
</form>

{% if result %}
  <div class="stats">
    <h3>✅ Count: {{ result.count }} trees</h3>
    <p>(Confidence Threshold: {{ result.threshold }})</p>
  </div>
  <img src="/result_image" />
{% endif %}
"""

app = Flask(__name__)
_app_state = {
    "image_bytes": None,
}

# --- CONFIGURATION ---
# Path to your best trained model. Update if yours is different.
DEFAULT_WEIGHTS = os.environ.get("WEIGHTS", r"runs/detect/yolo_tree_mixed3/weights/best.pt")
DEFAULT_CONF = float(os.environ.get("CONF", "0.25"))

# Initialize YOLO Model
print(f"Loading YOLO model from: {DEFAULT_WEIGHTS}...")
try:
    model = YOLO(DEFAULT_WEIGHTS)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"❌ Error loading model: {e}")
    print("Make sure the path to 'best.pt' is correct.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if model is None:
        return f"<h3>Error: Model not found at <code>{DEFAULT_WEIGHTS}</code>.</h3><p>Please check the file path in app_yolo.py</p>", 500

    conf_threshold = DEFAULT_CONF
    result = None

    if request.method == 'POST':
        # Get threshold from form
        try:
            conf_threshold = float(request.form.get('threshold', conf_threshold))
        except:
            pass

        if 'file' not in request.files or request.files['file'].filename == '':
            return render_template_string(HTML, result=None, default_threshold=conf_threshold)

        file = request.files['file']
        
        # 1. Open Image
        img = Image.open(file.stream).convert('RGB')

        # 2. Run Inference
        # conf=conf_threshold tells YOLO to ignore weak detections
        results = model.predict(img, conf=conf_threshold)
        
        # 3. Process Results
        r = results[0]
        count = len(r.boxes) # Count the number of boxes found

        # 4. Generate Visualization
        # r.plot() returns a NumPy array in BGR (Blue-Green-Red) format
        res_plotted = r.plot()
        
        # Convert BGR back to RGB for PIL to save correctly
        img_vis = Image.fromarray(res_plotted[..., ::-1]) 

        # Save to memory buffer to serve via web
        buf = io.BytesIO()
        img_vis.save(buf, format='PNG')
        buf.seek(0)
        _app_state["image_bytes"] = buf.read()

        result = {"count": count, "threshold": conf_threshold}

    return render_template_string(HTML, result=result, default_threshold=DEFAULT_CONF)

@app.route('/result_image')
def result_image():
    if _app_state["image_bytes"] is None:
        return "No image processed yet", 404
    return send_file(io.BytesIO(_app_state["image_bytes"]), mimetype='image/png')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", "5000"))
    print(f"Starting server on http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)