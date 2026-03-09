import argparse
import subprocess
import sys
from pathlib import Path


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


def infer_yolo_version_from_weights(weights_path: str) -> str | None:
    normalized = str(weights_path).lower()
    if "yolov8" in normalized:
        return "yolov8"
    if "yolov5" in normalized:
        return "yolov5"
    return None


def resolve_yolo_selection(requested_version: str, weights_path: str) -> tuple[str, str]:
    inferred_version = infer_yolo_version_from_weights(weights_path)
    resolved_version = inferred_version or requested_version

    if requested_version == "yolov8" and Path(weights_path).name == "yolov5s.pt":
        return "yolov8", str(ROOT / "yolov8n.pt")
    if resolved_version != requested_version:
        print(
            "Detector/version mismatch detected. "
            f"weights={Path(weights_path).name} requested={requested_version} "
            f"resolved={resolved_version}"
        )
    return resolved_version, weights_path


def run_cmd(cmd):
    print(f"$ {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit code {rc}")


def main():
    parser = argparse.ArgumentParser(
        description="Run tracking + CLEAR MOT evaluation + chart generation in one command."
    )
    parser.add_argument("--source", type=str, required=True, help="Video source for tracking.")
    parser.add_argument("--gt", type=Path, required=True, help="Ground-truth MOT txt.")
    parser.add_argument("--yolo-version", type=str, default="yolov5", choices=["yolov5", "yolov8"])
    parser.add_argument("--yolo-weights", type=str, default=str(ROOT / "yolov5s.pt"))
    parser.add_argument("--reid-weights", type=str, default=str(ROOT / "osnet_x0_25_msmt17.pt"))
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLOv8 inference size.")
    parser.add_argument("--max-det", type=int, default=150, help="YOLOv8 max detections per frame.")
    parser.add_argument("--person-only", action="store_true", dest="person_only", help="YOLOv8 only keeps class 0(person).")
    parser.add_argument("--all-classes", action="store_false", dest="person_only", help="YOLOv8 keeps all classes.")
    parser.add_argument("--eval-iou-thr", type=float, default=0.5, help="IoU threshold for CLEAR MOT matching.")
    parser.add_argument("--gt-class-id", type=int, default=1, help="GT class id filter (pedestrian=1).")
    parser.add_argument("--show", action="store_true", help="Show OpenCV preview during tracking.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("runs/clear_mot_pipeline"),
        help="Pipeline output directory.",
    )
    parser.set_defaults(person_only=True)
    args = parser.parse_args()
    args.yolo_version, args.yolo_weights = resolve_yolo_selection(
        args.yolo_version,
        args.yolo_weights,
    )

    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    video_path = run_dir / "tracked.mp4"
    mot_path = run_dir / "track_mot.txt"
    runtime_path = run_dir / "runtime.json"
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    track_cmd = [
        sys.executable,
        str(ROOT / "examples" / "track.py"),
        "--yolo-version",
        args.yolo_version,
        "--source",
        args.source,
        "--yolo-weights",
        args.yolo_weights,
        "--reid-weights",
        args.reid_weights,
        "--conf-thres",
        str(args.conf_thres),
        "--iou-thres",
        str(args.iou_thres),
        "--imgsz",
        str(args.imgsz),
        "--max-det",
        str(args.max_det),
        "--output",
        str(video_path),
        "--save-mot-txt",
        str(mot_path),
        "--runtime-json",
        str(runtime_path),
    ]
    if args.person_only:
        track_cmd.append("--person-only")
    else:
        track_cmd.append("--all-classes")
    if args.show:
        track_cmd.append("--show")
    run_cmd(track_cmd)

    eval_cmd = [
        sys.executable,
        str(ROOT / "examples" / "eval_clear_mot.py"),
        "--gt",
        str(args.gt),
        "--pred",
        str(mot_path),
        "--output-dir",
        str(eval_dir),
        "--runtime-json",
        str(runtime_path),
        "--iou-thr",
        str(args.eval_iou_thr),
        "--gt-class-id",
        str(args.gt_class_id),
    ]
    run_cmd(eval_cmd)

    print("")
    print("Pipeline finished.")
    print(f"Tracking video : {video_path}")
    print(f"MOT txt       : {mot_path}")
    print(f"Summary json  : {eval_dir / 'summary.json'}")
    print(f"Chart csv     : {eval_dir / 'chart_data.csv'}")
    print(f"Bar chart svg : {eval_dir / 'metrics_bar.svg'}")
    print(f"MOTA curve svg: {eval_dir / 'mota_curve.svg'}")


if __name__ == "__main__":
    main()
