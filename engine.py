from typing import Dict, Iterable, Tuple, List
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, data_loader: Iterable, device: torch.device, epoch: int, scaler: GradScaler, max_norm: float = 0.0) -> Dict[str, float]:
	model.train()
	loss_avg = 0.0
	pbar = tqdm(data_loader, desc=f"Epoch {epoch} [train]", ncols=100)
	for images, targets in pbar:
		images = [img.to(device) for img in images]
		targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

		with autocast(enabled=True):
			loss_dict = model(images, targets)
			loss = sum(loss for loss in loss_dict.values())

		scaler.scale(loss).backward()
		if max_norm and max_norm > 0:
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
		scaler.step(optimizer)
		scaler.update()
		optimizer.zero_grad(set_to_none=True)

		loss_avg = loss_avg * 0.9 + loss.item() * 0.1 if loss_avg > 0 else loss.item()
		pbar.set_postfix({"loss": f"{loss_avg:.4f}"})

	return {"loss": loss_avg}


def evaluate_count(model: nn.Module, data_loader: Iterable, device: torch.device, score_thresh: float = 0.5) -> Tuple[float, float, float]:
	model.eval()
	total_images = 0
	sum_abs_err = 0.0
	exact_match = 0
	with torch.no_grad():
		for images, targets in tqdm(data_loader, desc="Eval", ncols=100):
			images = [img.to(device) for img in images]
			outputs = model(images)
			for out, (_, target) in zip(outputs, zip(images, targets)):
				pred_count = int((out["scores"] >= score_thresh).sum().item())
				true_count = int(target["labels"].numel())
				sum_abs_err += abs(pred_count - true_count)
				if pred_count == true_count:
					exact_match += 1
				total_images += 1
	mae = sum_abs_err / max(1, total_images)
	accuracy = (exact_match / max(1, total_images)) * 100.0
	return mae, total_images, accuracy


def evaluate_count_sweep(model: nn.Module, data_loader: Iterable, device: torch.device, thresholds: List[float]) -> Tuple[float, float, float]:
	best_acc = -1.0
	best_mae = 1e9
	best_t = thresholds[0] if thresholds else 0.5
	for t in thresholds:
		mae, n, acc = evaluate_count(model, data_loader, device, score_thresh=t)
		# prefer higher accuracy; tie-break on lower MAE
		if acc > best_acc or (acc == best_acc and mae < best_mae):
			best_acc = acc
			best_mae = mae
			best_t = t
	return best_mae, best_acc, best_t


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, out_dir: str, filename: str = None):
	Path(out_dir).mkdir(parents=True, exist_ok=True)
	ckpt = {
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"epoch": epoch,
	}
	name = filename or f"checkpoint_epoch_{epoch}.pth"
	torch.save(ckpt, str(Path(out_dir) / name))


def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, ckpt_path: str) -> int:
	ckpt = torch.load(ckpt_path, map_location="cpu")
	model.load_state_dict(ckpt["model"])  # type: ignore
	if optimizer is not None and "optimizer" in ckpt:
		optimizer.load_state_dict(ckpt["optimizer"])  # type: ignore
	return int(ckpt.get("epoch", 0))
