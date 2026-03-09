# MOT16 + YOLOv8 + DeepSORT Workflow

This document summarizes a practical workflow for preparing MOT16 data, training a YOLOv8 detector, running DeepSORT tracking, and evaluating results with the scripts in this repository.

## 1. Prepare the Dataset

Place the original MOT16 dataset under:

```text
datasets/MOT16
```

Then convert MOT16 annotations into YOLO detection format:

```bash
python tools/prepare_mot16_yolo.py \
  --mot16-root datasets/MOT16 \
  --out datasets/mot16_yolo
```

Expected output:

- `datasets/mot16_yolo/images`
- `datasets/mot16_yolo/labels`
- `datasets/mot16_yolo/mot16.yaml`

## 2. Train a YOLOv8 Detector

Train a YOLOv8 model on the converted dataset:

```bash
python tools/train_yolov8_mot16.py \
  --data datasets/mot16_yolo/mot16.yaml \
  --model yolov8n.pt \
  --epochs 50
```

The trained weight is typically written to:

```text
runs/train/yolov8_mot16/weights/best.pt
```

## 3. Run Tracking from the CLI

Use the trained detector with DeepSORT:

```bash
python examples/track.py \
  --yolo-version yolov8 \
  --yolo-weights runs/train/yolov8_mot16/weights/best.pt \
  --reid-weights osnet_x0_25_msmt17.pt \
  --source test_video.mp4 \
  --show
```

Typical outputs include:

- a tracked video under `runs/`
- MOTChallenge-format tracking text
- optional runtime statistics JSON

## 4. Run the Evaluation Pipeline

If you already have a ground-truth MOT text file, run the full pipeline:

```bash
python examples/run_clear_mot_pipeline.py \
  --source test_video.mp4 \
  --gt /path/to/gt.txt \
  --yolo-version yolov8 \
  --yolo-weights runs/train/yolov8_mot16/weights/best.pt \
  --reid-weights osnet_x0_25_msmt17.pt \
  --show
```

This pipeline produces:

- tracked video output
- MOT prediction text
- `summary.json`
- `chart_data.csv`
- evaluation charts in SVG format

## 5. Use the GUI

Launch the GUI with:

```bash
python examples/gui.py
```

The GUI supports:

- YOLOv5 / YOLOv8 switching
- detector and ReID weight selection
- live tracking preview
- one-click evaluation pipeline execution
- one-click cleanup of tracking results
