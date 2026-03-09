import numpy as np
from ultralytics import YOLO


class YOLOv8:
    def __init__(
        self,
        weights,
        device,
        conf_thres=0.5,
        iou_thres=0.5,
        person_only=True,
        imgsz=640,
        max_det=300,
    ):
        self.device = getattr(device, "type", str(device))
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.person_only = person_only
        self.imgsz = imgsz
        self.max_det = max_det
        self.model = YOLO(weights)

    def get_dets(self, img):
        classes = [0] if self.person_only else None
        results = self.model.predict(
            source=img,
            device=self.device,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=classes,
            verbose=False,
        )
        if not results:
            return np.empty((0, 6), dtype=np.float32)

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.empty((0, 6), dtype=np.float32)

        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
        cls = r.boxes.cls.cpu().numpy().reshape(-1, 1)
        dets = np.concatenate([xyxy, conf, cls], axis=1).astype(np.float32)
        return dets
