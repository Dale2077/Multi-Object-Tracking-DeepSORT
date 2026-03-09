import argparse
from pathlib import Path

from ultralytics import YOLO


def train_yolov8_on_mot16(
    data_yaml: Path,
    model: str = "yolov8n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: Path = Path("runs/train"),
    name: str = "yolov8_mot16",
    device: str = "",
) -> Path:
    detector = YOLO(model)
    detector.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=str(project),
        name=name,
        device=device if device else None,
        pretrained=True,
    )
    best = project / name / "weights" / "best.pt"
    return best


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector on MOT16 YOLO-format dataset.")
    parser.add_argument("--data", type=Path, default=Path("datasets/mot16_yolo/mot16.yaml"))
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", type=str, default="yolov8_mot16")
    parser.add_argument("--device", type=str, default="")
    args = parser.parse_args()

    best = train_yolov8_on_mot16(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    print(f"Training done. Best weights: {best}")


if __name__ == "__main__":
    main()
