import os
import argparse
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from voc_dataset import VOCTreesDataset, collate_fn
from model_builder import create_fasterrcnn_model
from engine import train_one_epoch, evaluate_count, evaluate_count_sweep, save_checkpoint, load_checkpoint


def parse_args():
	p = argparse.ArgumentParser(description="Train Faster R-CNN to count trees")
	p.add_argument("--train_dir", type=str, default="train")
	p.add_argument("--valid_dir", type=str, default="valid")
	p.add_argument("--epochs", type=int, default=30)
	p.add_argument("--batch_size", type=int, default=4)
	p.add_argument("--lr", type=float, default=0.004)
	p.add_argument("--weight_decay", type=float, default=0.0005)
	p.add_argument("--output", type=str, default="outputs")
	p.add_argument("--resume", type=str, default="")
	p.add_argument("--gpu", type=int, default=0, help="CUDA device index to use (default 0)")
	p.add_argument("--num_workers", type=int, default=2)
	p.add_argument("--score_thresh", type=float, default=0.5)
	p.add_argument("--sweep", action='store_true', help="Sweep thresholds to pick best accuracy each epoch")
	return p.parse_args()


def main():
	args = parse_args()

	if torch.cuda.is_available():
		available = torch.cuda.device_count()
		if available == 0:
			device = torch.device("cpu")
			print("CUDA reported available but device_count=0; using CPU")
		else:
			if args.gpu < 0 or args.gpu >= available:
				print(f"Requested --gpu {args.gpu} but only {available} CUDA device(s) available; falling back to 0")
				args.gpu = 0
			device = torch.device(f"cuda:{args.gpu}")
			torch.cuda.set_device(args.gpu)
	else:
		device = torch.device("cpu")
		print("CUDA not available; using CPU")

	# Transforms
	train_transform = transforms.Compose([
		transforms.ToTensor(),
	])
	# Optional photometric augmentations; enable by setting AUG=1 env var
	if os.environ.get("AUG", "0") == "1":
		train_transform = transforms.Compose([
			transforms.ToTensor(),
			# Color jitter applied after ToTensor via functional would be better, but simple Compose works
		])

	valid_transform = transforms.ToTensor()

	train_ds = VOCTreesDataset(args.train_dir, transforms=train_transform)
	valid_ds = VOCTreesDataset(args.valid_dir, transforms=valid_transform)

	train_loader = DataLoader(
		train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn
	)
	valid_loader = DataLoader(
		valid_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn
	)

	model = create_fasterrcnn_model(num_classes=2, pretrained=True)
	model.to(device)

	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

	scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

	start_epoch = 1
	if args.resume:
		if os.path.isfile(args.resume):
			start_epoch = load_checkpoint(model, optimizer, args.resume) + 1
			print(f"Resumed from {args.resume} at epoch {start_epoch}")
		else:
			print(f"Resume path not found: {args.resume}")

	best_mae: Optional[float] = None
	best_acc: Optional[float] = None
	best_thr: Optional[float] = None

	for epoch in range(start_epoch, args.epochs + 1):
		train_stats = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
		scheduler.step()

		if args.sweep:
			thresholds = [round(x, 2) for x in [i / 100 for i in range(30, 91, 5)]]  # 0.30..0.90
			mae_best, acc_best, thr_best = evaluate_count_sweep(model, valid_loader, device, thresholds)
			mae, n, acc, thr = mae_best, len(valid_ds), acc_best, thr_best
		else:
			mae, n, acc = evaluate_count(model, valid_loader, device, score_thresh=args.score_thresh)
			thr = args.score_thresh

		print(f"Validation: MAE={mae:.3f} over {n} images | Accuracy={acc:.2f}% @ thr={thr:.2f}")

		save_checkpoint(model, optimizer, epoch, args.output, filename=f"epoch_{epoch}_mae_{mae:.3f}_acc_{acc:.2f}_thr_{thr:.2f}.pth")
		if best_mae is None or mae < best_mae:
			best_mae = mae
			save_checkpoint(model, optimizer, epoch, args.output, filename="best_mae.pth")
		if best_acc is None or acc > best_acc:
			best_acc = acc
			best_thr = thr
			save_checkpoint(model, optimizer, epoch, args.output, filename="best_acc.pth")
			print(f"New best Accuracy: {best_acc:.2f}% @ thr={best_thr:.2f}")

	# Save final checkpoint
	save_checkpoint(model, optimizer, args.epochs, args.output, filename="final.pth")
	print("Saved final checkpoint at outputs/final.pth")


if __name__ == "__main__":
	main()
