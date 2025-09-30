import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model_builder import create_fasterrcnn_model


def parse_args():
	p = argparse.ArgumentParser(description="Infer tree count on an image or directory")
	p.add_argument("--weights", type=str, required=True)
	p.add_argument("--input", type=str, required=True, help="Image file or directory")
	p.add_argument("--gpu", type=int, default=0, help="CUDA device index to use (default 0)")
	p.add_argument("--score_thresh", type=float, default=0.5)
	return p.parse_args()


def load_model(weights_path: str, device: torch.device):
	model = create_fasterrcnn_model(num_classes=2, pretrained=False)
	ckpt = torch.load(weights_path, map_location="cpu")
	state = ckpt.get("model", ckpt)
	model.load_state_dict(state)
	model.to(device)
	model.eval()
	return model


def main():
	args = parse_args()
	if torch.cuda.is_available() and torch.cuda.device_count() > 0:
		available = torch.cuda.device_count()
		if args.gpu < 0 or args.gpu >= available:
			print(f"Requested --gpu {args.gpu} but only {available} CUDA device(s) available; falling back to 0")
			args.gpu = 0
		device = torch.device(f"cuda:{args.gpu}")
		torch.cuda.set_device(args.gpu)
	else:
		device = torch.device("cpu")
		print("CUDA not available; using CPU")

	model = load_model(args.weights, device)
	to_tensor = transforms.ToTensor()

	inp = Path(args.input)
	paths = []
	if inp.is_dir():
		paths = [p for p in inp.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
	else:
		paths = [inp]

	for p in paths:
		img = Image.open(p).convert("RGB")
		tensor = to_tensor(img).to(device)
		with torch.no_grad():
			out = model([tensor])[0]
			scores = out["scores"].detach().cpu()
			count = int((scores >= args.score_thresh).sum().item())
		print(f"{p.name}: {count} trees (threshold={args.score_thresh})")


if __name__ == "__main__":
	main()
