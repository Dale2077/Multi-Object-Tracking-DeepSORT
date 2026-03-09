from pathlib import Path

import torch
import numpy as np


class YOLOv5:
    def __init__(self, weights, device, conf_thres=0.5, iou_thres=0.5):
        weights_path = Path(weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLOv5 weights not found: {weights_path}")
        if "yolov8" in weights_path.name.lower():
            raise ValueError(
                f"Incompatible weights for YOLOv5 detector: {weights_path.name}. "
                "Use the YOLOv8 detector for YOLOv8 weights."
            )

        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        try:
            self.model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(weights_path),
                device=device,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load YOLOv5 weights via torch.hub. "
                f"weights={weights_path} error={e}"
            ) from e

        self.model.conf = conf_thres
        self.model.iou = iou_thres
        self.model.classes = [0]

    def get_dets(self, img):
        """
        Run inference on a BGR frame and return detections as
        ``[x1, y1, x2, y2, conf, class_id]``.
        """
        results = self.model(img)
        preds = results.xyxy[0].cpu().numpy()

        return preds
