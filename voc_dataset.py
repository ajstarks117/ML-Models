import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


CLASS_NAME = "trees"


def _read_xml_boxes(xml_path: str) -> Tuple[np.ndarray, List[int]]:
	root = ET.parse(xml_path).getroot()
	boxes: List[List[float]] = []
	labels: List[int] = []
	for obj in root.findall("object"):
		name = obj.find("name").text
		if name is None:
			continue
		if name.strip().lower() != CLASS_NAME:
			# skip unknown classes if any
			continue
		bnd = obj.find("bndbox")
		xmin = float(bnd.find("xmin").text)
		ymin = float(bnd.find("ymin").text)
		xmax = float(bnd.find("xmax").text)
		ymax = float(bnd.find("ymax").text)
		boxes.append([xmin, ymin, xmax, ymax])
		labels.append(1)  # 1 for tree, 0 reserved for background
	if len(boxes) == 0:
		return np.zeros((0, 4), dtype=np.float32), []
	return np.asarray(boxes, dtype=np.float32), labels


class VOCTreesDataset(Dataset):
	def __init__(self, root: str, transforms: Any = None) -> None:
		self.root = root
		self.transforms = transforms
		# pair images and xmls
		all_files = sorted(os.listdir(root))
		self.images = [f for f in all_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
		self.ann_map: Dict[str, str] = {}
		for img in self.images:
			stem = os.path.splitext(img)[0]
			xml = stem + ".xml"
			xml_path = os.path.join(root, xml)
			if os.path.exists(xml_path):
				self.ann_map[img] = xml
			else:
				# keep image, but it will have zero targets
				self.ann_map[img] = None

	def __len__(self) -> int:
		return len(self.images)

	def __getitem__(self, idx: int) -> Tuple[Image.Image, Dict[str, torch.Tensor]]:
		img_name = self.images[idx]
		img_path = os.path.join(self.root, img_name)
		img = Image.open(img_path).convert("RGB")

		xml_name = self.ann_map.get(img_name)
		if xml_name is not None:
			boxes_np, labels_list = _read_xml_boxes(os.path.join(self.root, xml_name))
		else:
			boxes_np, labels_list = np.zeros((0, 4), dtype=np.float32), []

		boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
		labels = torch.as_tensor(labels_list, dtype=torch.int64)
		area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
		iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

		target: Dict[str, torch.Tensor] = {
			"boxes": boxes,
			"labels": labels,
			"image_id": torch.tensor([idx], dtype=torch.int64),
			"area": area,
			"iscrowd": iscrowd,
		}

		if self.transforms is not None:
			img = self.transforms(img)

		return img, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
	return tuple(zip(*batch))
