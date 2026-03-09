import argparse
import cv2
import torch
import sys
import os
import json
import time
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from pathlib import Path
import numpy as np

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from boxmot.trackers.tracker_zoo import create_tracker
from boxmot.detectors.yolov5.yolov5 import YOLOv5
from boxmot.detectors.yolov8.yolov8 import YOLOv8


def _infer_yolo_version_from_weights(weights_path: str) -> str | None:
    """Infer the detector family from a weight filename when possible."""
    normalized = str(weights_path).lower()
    if "yolov8" in normalized:
        return "yolov8"
    if "yolov5" in normalized:
        return "yolov5"
    return None


def _resolve_yolo_version(requested_version: str, weights_path: str) -> str:
    """Prefer an explicit version unless the weight filename clearly disagrees."""
    inferred_version = _infer_yolo_version_from_weights(weights_path)
    if inferred_version and inferred_version != requested_version:
        print(
            "Detector/version mismatch detected. "
            f"weights={Path(weights_path).name} requested={requested_version} "
            f"resolved={inferred_version}"
        )
        return inferred_version
    return requested_version


def _write_mot_line(file_obj, frame_id, track_output):
    """Write one tracking result row in MOTChallenge text format."""
    x1, y1, x2, y2, track_id, conf, cls, _ = track_output
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    class_id = int(cls) + 1
    file_obj.write(
        f"{int(frame_id)},{int(track_id)},{x1:.3f},{y1:.3f},{w:.3f},{h:.3f},1,{class_id},{float(conf):.6f}\n"
    )


def run(args):
    """Run detection, tracking, visualization, and result export on one source."""
    yolo_version = _resolve_yolo_version(args.yolo_version, str(args.yolo_weights))

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple M2 (MPS) acceleration activated.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("Warning: running on CPU. Performance may be limited.")

    print(f"Loading {yolo_version.upper()} model from {args.yolo_weights}...")
    if yolo_version == "yolov8":
        model = YOLOv8(
            weights=args.yolo_weights,
            device=device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            person_only=args.person_only,
            imgsz=args.imgsz,
            max_det=args.max_det,
        )
    else:
        model = YOLOv5(
            weights=args.yolo_weights,
            device=device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
        )

    print("Loading DeepSORT tracker...")
    tracker = create_tracker(
        tracker_type='deepsort',
        tracker_config=ROOT / 'boxmot/configs/deepsort.yaml',
        reid_weights=args.reid_weights,
        device=device,
        half=False,
        per_class=False,
    )

    source = args.source
    if source.isdigit():
        vid = cv2.VideoCapture(int(source))
        print("Opening webcam...")
    else:
        vid = cv2.VideoCapture(source)
        print(f"Opening video: {source}")

    if not vid.isOpened():
        print(f"Error: cannot open video source {source}")
        return

    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    print(f"Processing... Output will be saved to {save_path}")

    mot_file = None
    if args.save_mot_txt:
        mot_path = Path(args.save_mot_txt)
        mot_path.parent.mkdir(parents=True, exist_ok=True)
        mot_file = open(mot_path, "w", encoding="utf-8")
        print(f"MOT result txt will be saved to {mot_path}")

    frame_count = 0
    t0 = time.perf_counter()
    while True:
        ret, im = vid.read()
        if not ret:
            break

        frame_count += 1

        dets = model.get_dets(im)
        tracker_outputs = tracker.update(dets, im)

        for output in tracker_outputs:
            x1, y1, x2, y2, id, conf, cls, ind = output

            cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            label = f"ID: {int(id)}"
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.rectangle(im, (int(x1), int(y1)-t_size[1]-5), (int(x1)+t_size[0], int(y1)), (0,0,255), -1)
            cv2.putText(im, label, (int(x1), int(y1)-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if mot_file is not None:
                _write_mot_line(mot_file, frame_count, output)

        if args.show:
            cv2.imshow('DeepSORT Tracking', im)
            if cv2.waitKey(1) == ord('q'):
                break

        out.write(im)

        if frame_count % 20 == 0:
            print(f"Processing frame {frame_count}...")

    vid.release()
    out.release()
    if mot_file is not None:
        mot_file.close()
    cv2.destroyAllWindows()
    elapsed_sec = max(time.perf_counter() - t0, 1e-9)
    runtime_fps = frame_count / elapsed_sec
    print("Done!")
    print(f"Total frames: {frame_count}")
    print(f"Elapsed time: {elapsed_sec:.3f}s")
    print(f"End-to-end FPS: {runtime_fps:.3f}")

    runtime_stats = {
        "frames": frame_count,
        "elapsed_sec": elapsed_sec,
        "fps": runtime_fps,
        "yolo_version": yolo_version,
        "yolo_weights": str(args.yolo_weights),
        "source": str(args.source),
        "output_video": str(save_path),
        "output_mot_txt": str(args.save_mot_txt) if args.save_mot_txt else "",
    }

    if args.runtime_json:
        runtime_path = Path(args.runtime_json)
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        with open(runtime_path, "w", encoding="utf-8") as f:
            json.dump(runtime_stats, f, ensure_ascii=False, indent=2)
        print(f"Runtime stats saved to {runtime_path}")

    return runtime_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-version', type=str, default='yolov5', choices=['yolov5', 'yolov8'], help='detector version')
    parser.add_argument('--yolo-weights', type=str, default=ROOT / 'yolov5s.pt', help='model.pt path')
    parser.add_argument('--reid-weights', type=str, default=ROOT / 'osnet_x0_25_msmt17.pt', help='reid model path')
    parser.add_argument('--source', type=str, default='0', help='source')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--show', action='store_true', help='visualize results')
    parser.add_argument('--output', type=str, default=str(ROOT / 'output.mp4'), help='output video path')
    parser.add_argument('--save-mot-txt', type=str, default='', help='save tracking result in MOT txt format')
    parser.add_argument('--runtime-json', type=str, default='', help='save runtime stats json file')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLOv8 inference size')
    parser.add_argument('--max-det', type=int, default=150, help='YOLOv8 max detections per frame')
    parser.add_argument('--person-only', action='store_true', dest='person_only', help='YOLOv8 only keeps class 0(person)')
    parser.add_argument('--all-classes', action='store_false', dest='person_only', help='YOLOv8 keeps all classes')
    parser.set_defaults(person_only=True)
    
    args = parser.parse_args()
    run(args)
