from typing import Optional
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_fasterrcnn_model(num_classes: int = 2, pretrained: bool = True):
	# Faster R-CNN with ResNet50-FPN v2 backbone
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=(
		torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
	))
	in_features = model.roi_heads.box_predictor.cls_score.in_features
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	return model
