"""Post-hoc failure analysis for the held-out video benchmark.

Loads the FP32 benchmark CSV and picks the worst frames by two signals:
  - highest-confidence detection of any kind (candidate false positives if actually background)
  - frames near the confidence threshold (borderline)

For each pick, extracts the raw frame and the same frame with predicted boxes
drawn on it, so the failure can be visually reviewed.

Usage:
    python failure_analysis.py \
        --csv runs/bench/fp32.csv \
        --video test_video.mp4 \
        --model runs/detect/defect_v1/weights/best.onnx \
        --out runs/bench/failures
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from infer import Detector


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "frame": int(row["frame"]),
                "num_detections": int(row["num_detections"]),
                "mean_confidence": float(row["mean_confidence"]),
                "latency_ms": float(row["latency_ms"]),
            })
    return rows


def pick_failures(rows, n=3):
    """Highest-confidence frames plus borderline frames near threshold."""
    with_dets = [r for r in rows if r["num_detections"] > 0]
    with_dets.sort(key=lambda r: r["mean_confidence"], reverse=True)
    top_conf = with_dets[:n]
    borderline = sorted(
        (r for r in with_dets if 0.25 <= r["mean_confidence"] <= 0.4),
        key=lambda r: r["mean_confidence"],
    )[:n]
    return top_conf, borderline


def dump_frames(video, model, frames_to_dump, out_dir, conf, iou):
    out_dir.mkdir(parents=True, exist_ok=True)
    det = Detector(str(model), conf=conf, iou=iou)
    cap = cv2.VideoCapture(str(video))
    target = {r["frame"]: r for r in frames_to_dump}
    if not target:
        cap.release()
        return
    max_frame = max(target)
    idx = 0
    while idx <= max_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in target:
            r = target[idx]
            raw_path = out_dir / f"frame_{idx:04d}_raw.jpg"
            cv2.imwrite(str(raw_path), frame)
            boxes, scores, _ = det(frame)
            annotated = frame.copy()
            for b, s in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{s:.2f}", (x1, max(0, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            ann_path = out_dir / f"frame_{idx:04d}_pred.jpg"
            cv2.imwrite(str(ann_path), annotated)
            print(f"  frame {idx:4d}  conf={r['mean_confidence']:.3f}  dets={r['num_detections']}  -> {ann_path.name}")
        idx += 1
    cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="runs/bench/failures")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    top_conf, borderline = pick_failures(rows, args.n)
    print(f"top-{args.n} highest-confidence frames:")
    for r in top_conf:
        print(f"  frame {r['frame']:4d}  conf={r['mean_confidence']:.3f}  dets={r['num_detections']}")
    print(f"top-{args.n} borderline (0.25-0.4) frames:")
    for r in borderline:
        print(f"  frame {r['frame']:4d}  conf={r['mean_confidence']:.3f}  dets={r['num_detections']}")

    dump_frames(Path(args.video), Path(args.model),
                top_conf + borderline, Path(args.out), args.conf, args.iou)


if __name__ == "__main__":
    main()
