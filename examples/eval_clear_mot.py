import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_mot_rows(path: Path) -> List[List[float]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file but got directory/path: {path}")
    if path.stat().st_size == 0:
        return []
    rows: List[List[float]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw:
                continue
            rows.append([float(x) for x in raw])
    return rows


def to_xyxy(row: List[float]) -> Tuple[float, float, float, float]:
    x = row[2]
    y = row[3]
    w = row[4]
    h = row[5]
    return x, y, x + w, y + h


def box_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def match_by_iou(gt_rows: List[List[float]], pred_rows: List[List[float]], threshold: float) -> List[Tuple[int, int]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt_rows):
        gb = to_xyxy(g)
        for pi, p in enumerate(pred_rows):
            pb = to_xyxy(p)
            iou = box_iou(gb, pb)
            if iou >= threshold:
                pairs.append((iou, gi, pi))
    pairs.sort(key=lambda x: x[0], reverse=True)

    used_g = set()
    used_p = set()
    matches: List[Tuple[int, int]] = []
    for _, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi))
    return matches


def parse_frames(
    gt_rows: List[List[float]],
    pred_rows: List[List[float]],
    gt_class_id: int,
    use_gt_visibility: bool,
) -> Tuple[Dict[int, List[List[float]]], Dict[int, List[List[float]]], int]:
    if not gt_rows:
        raise ValueError("GT file is empty.")
    if len(gt_rows[0]) < 6:
        raise ValueError("GT file must have at least 6 columns: frame,id,x,y,w,h.")
    if pred_rows and len(pred_rows[0]) < 6:
        raise ValueError("Prediction file must have at least 6 columns: frame,id,x,y,w,h.")

    filtered_gt: List[List[float]] = []
    for row in gt_rows:
        if use_gt_visibility and len(row) > 6 and row[6] <= 0:
            continue
        if gt_class_id > 0 and len(row) > 7 and int(row[7]) != gt_class_id:
            continue
        filtered_gt.append(row)

    gt_frames: Dict[int, List[List[float]]] = {}
    pred_frames: Dict[int, List[List[float]]] = {}
    max_frame = 0

    for row in filtered_gt:
        f = int(row[0])
        gt_frames.setdefault(f, []).append(row)
        if f > max_frame:
            max_frame = f

    for row in pred_rows:
        f = int(row[0])
        pred_frames.setdefault(f, []).append(row)
        if f > max_frame:
            max_frame = f

    return gt_frames, pred_frames, max_frame


def evaluate_clear_mot(
    gt_frames: Dict[int, List[List[float]]],
    pred_frames: Dict[int, List[List[float]]],
    max_frame: int,
    iou_thr: float,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0
    evaluated_frames = 0

    prev_match_gt_to_pred: Dict[int, int] = {}
    frame_records: List[Dict[str, float]] = []

    cum_tp = 0
    cum_fp = 0
    cum_fn = 0
    cum_idsw = 0
    cum_gt = 0

    for frame in range(1, max_frame + 1):
        gt_rows = gt_frames.get(frame, [])
        pred_rows = pred_frames.get(frame, [])
        matches = match_by_iou(gt_rows, pred_rows, iou_thr)

        tp = len(matches)
        fp = len(pred_rows) - tp
        fn = len(gt_rows) - tp

        idsw = 0
        curr_match_gt_to_pred: Dict[int, int] = {}
        for gi, pi in matches:
            gid = int(gt_rows[gi][1])
            pid = int(pred_rows[pi][1])
            curr_match_gt_to_pred[gid] = pid
            prev_pid = prev_match_gt_to_pred.get(gid)
            if prev_pid is not None and prev_pid != pid:
                idsw += 1
        prev_match_gt_to_pred = curr_match_gt_to_pred

        total_gt += len(gt_rows)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_idsw += idsw
        evaluated_frames += 1

        cum_tp += tp
        cum_fp += fp
        cum_fn += fn
        cum_idsw += idsw
        cum_gt += len(gt_rows)
        cum_mota = 1.0 - ((cum_fn + cum_fp + cum_idsw) / cum_gt) if cum_gt > 0 else 0.0

        frame_records.append(
            {
                "frame": float(frame),
                "gt": float(len(gt_rows)),
                "pred": float(len(pred_rows)),
                "tp": float(tp),
                "fp": float(fp),
                "fn": float(fn),
                "idsw": float(idsw),
                "cum_mota": float(cum_mota * 100.0),
            }
        )

    mota = 1.0 - ((total_fn + total_fp + total_idsw) / total_gt) if total_gt > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    summary = {
        "frames": float(evaluated_frames),
        "gt_total": float(total_gt),
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
        "idsw": float(total_idsw),
        "mota": float(mota * 100.0),
        "precision": float(precision * 100.0),
        "recall": float(recall * 100.0),
    }
    return summary, frame_records


def resolve_fps(frames: int, elapsed_sec: float, fallback_fps: float) -> float:
    if elapsed_sec > 0:
        return frames / elapsed_sec
    if fallback_fps > 0:
        return fallback_fps
    return 0.0


def write_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_runtime_elapsed_sec(runtime_json: Path) -> float:
    if not runtime_json.exists():
        raise FileNotFoundError(f"Runtime json not found: {runtime_json}")
    with open(runtime_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get("elapsed_sec", 0.0))


def write_metrics_bar_svg(path: Path, summary: Dict[str, float]) -> None:
    metrics = [
        ("MOTA", float(summary["mota"])),
        ("IDSW", float(summary["idsw"])),
        ("FPS", float(summary["fps"])),
    ]
    width = 900
    height = 520
    left = 110
    right = 40
    top = 60
    bottom = 80
    chart_w = width - left - right
    chart_h = height - top - bottom
    bar_w = 150
    gap = (chart_w - bar_w * len(metrics)) / (len(metrics) + 1)

    values = [m[1] for m in metrics]
    vmax = max(values) if values else 1.0
    vmax = max(vmax, 1.0)

    colors = {"MOTA": "#0F766E", "IDSW": "#B45309", "FPS": "#2563EB"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="40" y="36" font-size="26" fill="#111827" font-family="Arial" font-weight="700">CLEAR MOT Metrics</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#9CA3AF" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#9CA3AF" stroke-width="2"/>',
    ]

    for i, (name, value) in enumerate(metrics):
        x = left + gap * (i + 1) + bar_w * i
        h = (value / vmax) * chart_h if vmax > 0 else 0
        y = top + chart_h - h
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" fill="{colors[name]}" rx="8"/>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 10:.2f}" text-anchor="middle" font-size="18" fill="#111827" font-family="Arial">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.2f}" y="{top + chart_h + 35:.2f}" text-anchor="middle" font-size="20" fill="#111827" font-family="Arial">{name}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def write_mota_curve_svg(path: Path, frame_records: List[Dict[str, float]]) -> None:
    width = 1000
    height = 560
    left = 90
    right = 40
    top = 60
    bottom = 90
    chart_w = width - left - right
    chart_h = height - top - bottom

    if not frame_records:
        x_values = [1]
        y_values = [0.0]
    else:
        x_values = [int(r["frame"]) for r in frame_records]
        y_values = [float(r["cum_mota"]) for r in frame_records]

    x_min, x_max = min(x_values), max(x_values)
    if x_max == x_min:
        x_max = x_min + 1
    y_min = min(min(y_values), 0.0)
    y_max = max(max(y_values), 1.0)
    if y_max == y_min:
        y_max = y_min + 1.0

    def sx(x):
        return left + (x - x_min) / (x_max - x_min) * chart_w

    def sy(y):
        return top + chart_h - (y - y_min) / (y_max - y_min) * chart_h

    points = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in zip(x_values, y_values))
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="#F8FAFC"/>',
        '<text x="40" y="36" font-size="26" fill="#111827" font-family="Arial" font-weight="700">Cumulative MOTA by Frame</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#9CA3AF" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#9CA3AF" stroke-width="2"/>',
        f'<polyline fill="none" stroke="#0F766E" stroke-width="3" points="{points}"/>',
        f'<text x="{left + chart_w / 2:.2f}" y="{height - 25}" text-anchor="middle" font-size="18" fill="#374151" font-family="Arial">Frame</text>',
        f'<text transform="translate(24,{top + chart_h / 2:.2f}) rotate(-90)" text-anchor="middle" font-size="18" fill="#374151" font-family="Arial">MOTA (%)</text>',
    ]

    ticks = 5
    for i in range(ticks + 1):
        yv = y_min + (y_max - y_min) * i / ticks
        yy = sy(yv)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + chart_w}" y2="{yy:.2f}" stroke="#E5E7EB" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{yy + 5:.2f}" text-anchor="end" font-size="13" fill="#6B7280" font-family="Arial">{yv:.1f}</text>')

    xticks = min(8, len(x_values))
    for i in range(xticks):
        xv = x_min + (x_max - x_min) * i / max(1, xticks - 1)
        xx = sx(xv)
        parts.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + chart_h}" stroke="#E5E7EB" stroke-width="1"/>')
        parts.append(f'<text x="{xx:.2f}" y="{top + chart_h + 24:.2f}" text-anchor="middle" font-size="13" fill="#6B7280" font-family="Arial">{int(round(xv))}</text>')

    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="CLEAR MOT evaluator (MOTA / IDSW / FPS) with chart data export.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to MOT ground-truth txt.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to tracker output txt (MOT format).")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/clear_mot_eval"), help="Output folder.")
    parser.add_argument("--iou-thr", type=float, default=0.5, help="IoU threshold for frame association.")
    parser.add_argument("--gt-class-id", type=int, default=1, help="Filter GT class id (MOT pedestrian=1). <=0 disables filter.")
    parser.add_argument("--disable-gt-mark-filter", action="store_true", help="Disable GT mark/visibility filtering.")
    parser.add_argument("--elapsed-sec", type=float, default=0.0, help="End-to-end total runtime in seconds (for FPS).")
    parser.add_argument("--fps", type=float, default=0.0, help="Fallback FPS when elapsed-sec is not provided.")
    parser.add_argument("--runtime-json", type=Path, default=None, help="Runtime json from examples/track.py; used to read elapsed_sec.")
    args = parser.parse_args()

    gt_rows = load_mot_rows(args.gt)
    pred_rows = load_mot_rows(args.pred)
    gt_frames, pred_frames, max_frame = parse_frames(
        gt_rows=gt_rows,
        pred_rows=pred_rows,
        gt_class_id=args.gt_class_id,
        use_gt_visibility=not args.disable_gt_mark_filter,
    )

    summary, frame_records = evaluate_clear_mot(
        gt_frames=gt_frames,
        pred_frames=pred_frames,
        max_frame=max_frame,
        iou_thr=args.iou_thr,
    )

    elapsed_sec = args.elapsed_sec
    if args.runtime_json is not None and elapsed_sec <= 0:
        elapsed_sec = load_runtime_elapsed_sec(args.runtime_json)
    summary["fps"] = float(resolve_fps(int(summary["frames"]), elapsed_sec, args.fps))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    frame_csv_path = output_dir / "frame_metrics.csv"
    chart_csv_path = output_dir / "chart_data.csv"
    bar_svg_path = output_dir / "metrics_bar.svg"
    mota_svg_path = output_dir / "mota_curve.svg"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    write_csv(
        frame_csv_path,
        frame_records,
        ["frame", "gt", "pred", "tp", "fp", "fn", "idsw", "cum_mota"],
    )
    write_csv(
        chart_csv_path,
        [
            {"metric": "MOTA", "value": summary["mota"]},
            {"metric": "IDSW", "value": summary["idsw"]},
            {"metric": "FPS", "value": summary["fps"]},
        ],
        ["metric", "value"],
    )
    write_metrics_bar_svg(bar_svg_path, summary)
    write_mota_curve_svg(mota_svg_path, frame_records)

    print("CLEAR MOT evaluation done.")
    print(f"MOTA: {summary['mota']:.3f}")
    print(f"IDSW: {int(summary['idsw'])}")
    print(f"FPS : {summary['fps']:.3f}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {frame_csv_path}")
    print(f"Saved: {chart_csv_path}")
    print(f"Saved: {bar_svg_path}")
    print(f"Saved: {mota_svg_path}")


if __name__ == "__main__":
    main()
