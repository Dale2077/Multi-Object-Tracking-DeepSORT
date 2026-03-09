import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def read_seq_size(seq_dir: Path) -> Tuple[int, int]:
    seqinfo = seq_dir / "seqinfo.ini"
    w = h = None
    if seqinfo.exists():
        for line in seqinfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("imWidth="):
                w = int(line.split("=", 1)[1])
            elif line.startswith("imHeight="):
                h = int(line.split("=", 1)[1])
    if not w or not h:
        first_img = next((seq_dir / "img1").glob("*.jpg"), None)
        if first_img is None:
            raise FileNotFoundError(f"No images found in {seq_dir / 'img1'}")
        import cv2

        im = cv2.imread(str(first_img))
        h, w = im.shape[:2]
    return w, h


def parse_gt(gt_path: Path, class_id: int = 1) -> Dict[int, List[List[float]]]:
    out: Dict[int, List[List[float]]] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            frame = int(float(row[0]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            mark = int(float(row[6]))
            cls = int(float(row[7]))
            if mark != 1 or cls != class_id:
                continue
            out.setdefault(frame, []).append([x, y, w, h])
    return out


def to_yolo_label(box: List[float], img_w: int, img_h: int) -> str:
    x, y, w, h = box
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Convert MOT16 train split to YOLO detection dataset.")
    parser.add_argument("--mot16-root", type=Path, default=Path("datasets/MOT16"))
    parser.add_argument("--out", type=Path, default=Path("datasets/mot16_yolo"))
    parser.add_argument("--val-ratio", type=float, default=0.3)
    parser.add_argument("--class-id", type=int, default=1, help="MOT class id to export (person=1).")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlink.")
    args = parser.parse_args()

    train_root = args.mot16_root / "train"
    if not train_root.exists():
        raise FileNotFoundError(f"MOT16 train folder not found: {train_root}")

    out = args.out
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    seqs = sorted([p for p in train_root.iterdir() if p.is_dir()])
    n_val = max(1, int(len(seqs) * args.val_ratio))
    val_set = set([p.name for p in seqs[-n_val:]])

    for seq_dir in seqs:
        split = "val" if seq_dir.name in val_set else "train"
        img_w, img_h = read_seq_size(seq_dir)
        gt = parse_gt(seq_dir / "gt" / "gt.txt", class_id=args.class_id)
        img_dir = seq_dir / "img1"
        imgs = sorted(img_dir.glob("*.jpg"))

        for img_path in imgs:
            frame = int(img_path.stem)
            stem = f"{seq_dir.name}_{img_path.stem}"
            dst_img = out / "images" / split / f"{stem}.jpg"
            dst_lbl = out / "labels" / split / f"{stem}.txt"

            if not dst_img.exists():
                if args.copy_images:
                    shutil.copy2(img_path, dst_img)
                else:
                    dst_img.symlink_to(img_path.resolve())

            boxes = gt.get(frame, [])
            lines = [to_yolo_label(b, img_w, img_h) for b in boxes]
            dst_lbl.write_text("\n".join(lines), encoding="utf-8")

    yaml_path = out / "mot16.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {out.resolve()}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Prepared YOLO dataset at: {out}")
    print(f"Data yaml: {yaml_path}")


if __name__ == "__main__":
    main()
