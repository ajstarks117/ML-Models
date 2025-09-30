import io
import os
from pathlib import Path

from flask import Flask, request, render_template_string, send_file
import torch
from PIL import Image, ImageDraw
from torchvision import transforms

from model_builder import create_fasterrcnn_model


HTML = """
<!doctype html>
<title>Tree Counter</title>
<h2>Upload an image to detect and count trees</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept="image/*">
  <label>Threshold (0-1): <input type=number name=threshold step=0.05 min=0 max=1 value="{{ result.threshold if result else default_threshold }}"></label>
  <input type=submit value=Upload>
</form>
{% if result %}
  <h3>Result: {{ result.count }} trees (threshold={{ result.threshold }})</h3>
  {% if result.count == 0 and result.top_score is not none %}
    <p>Top score was {{ "%.2f"|format(result.top_score) }}. Try lowering threshold.</p>
  {% endif %}
  <img src="/result_image" style="max-width: 100%; height: auto;" />
{% endif %}
"""

app = Flask(__name__)
_app_state = {
	"image_bytes": None,
}

# Initialize model once
DEFAULT_WEIGHTS = os.environ.get("WEIGHTS", "outputs/final.pth")
DEFAULT_GPU = int(os.environ.get("GPU", "0"))
DEFAULT_THRESHOLD = float(os.environ.get("THRESH", "0.5"))

if torch.cuda.is_available() and torch.cuda.device_count() > 0:
	if DEFAULT_GPU < 0 or DEFAULT_GPU >= torch.cuda.device_count():
		DEFAULT_GPU = 0
	DEVICE = torch.device(f"cuda:{DEFAULT_GPU}")
	torch.cuda.set_device(DEFAULT_GPU)
else:
	DEVICE = torch.device("cpu")


def _load_model(weights_path: str, device: torch.device):
	model = create_fasterrcnn_model(num_classes=2, pretrained=False)
	ckpt = torch.load(weights_path, map_location="cpu")
	state = ckpt.get("model", ckpt)
	model.load_state_dict(state)
	model.to(device)
	model.eval()
	return model


MODEL = _load_model(DEFAULT_WEIGHTS, DEVICE) if Path(DEFAULT_WEIGHTS).exists() else None
TO_TENSOR = transforms.ToTensor()


def draw_detections(img: Image.Image, boxes, scores, threshold: float = 0.5) -> Image.Image:
	draw = ImageDraw.Draw(img)
	for box, score in zip(boxes, scores):
		if float(score) < threshold:
			continue
		x1, y1, x2, y2 = [float(v) for v in box]
		draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
		draw.text((x1, y1), f"{score:.2f}", fill=(255, 0, 0))
	return img


@app.route('/', methods=['GET', 'POST'])
def index():
	if MODEL is None:
		return "Model weights not found. Set WEIGHTS env var to a valid .pth and restart.", 500

	threshold = DEFAULT_THRESHOLD
	if request.method == 'POST':
		try:
			threshold = float(request.form.get('threshold', threshold))
		except Exception:
			pass

	result = None
	if request.method == 'POST':
		if 'file' not in request.files:
			return render_template_string(HTML, result=None, default_threshold=threshold)
		file = request.files['file']
		if file.filename == '':
			return render_template_string(HTML, result=None, default_threshold=threshold)

		img = Image.open(file.stream).convert('RGB')
		tensor = TO_TENSOR(img).to(DEVICE)
		with torch.no_grad():
			out = MODEL([tensor])[0]
			scores = out["scores"].detach().cpu()
			boxes = out["boxes"].detach().cpu()
			count = int((scores >= threshold).sum().item())
			top_score = float(scores.max().item()) if scores.numel() > 0 else None

		img_vis = draw_detections(img.copy(), boxes, scores, threshold)
		buf = io.BytesIO()
		img_vis.save(buf, format='PNG')
		buf.seek(0)
		_app_state["image_bytes"] = buf.read()

		result = {"count": count, "threshold": threshold, "top_score": top_score}

	return render_template_string(HTML, result=result, default_threshold=DEFAULT_THRESHOLD)


@app.route('/result_image')
def result_image():
	if _app_state["image_bytes"] is None:
		return "No image processed yet", 404
	return send_file(io.BytesIO(_app_state["image_bytes"]), mimetype='image/png')


if __name__ == '__main__':
	port = int(os.environ.get("PORT", "5000"))
	app.run(host='0.0.0.0', port=port, debug=True)
