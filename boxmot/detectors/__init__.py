from boxmot.detectors.detector import Detector
from .yolov5.yolov5 import YOLOv5
from .yolov8.yolov8 import YOLOv8

__all__ = [
    "Detector",
    "YOLOv5",
    "YOLOv8",
]
