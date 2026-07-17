"""Benchmark FP32 / FP16 / INT8 ONNX exports on a video source.

For each model that exists in the weights dir, walks the video frame by frame
using infer.Detector, logs per-frame CSV (frame, num_detections, mean_confidence,
latency_ms), and prints a summary table (size, mean/p95 latency, total detections).

Usage:
    python bench.py --weights-dir runs/detect/defect_v1/weights --video test_video.mp4
"""

import argparse
import csv
from pathlib import Path

import cv2

from infer import Detector


def bench_one(model_path, video, conf, iou, out_csv):
    det = Detector(str(model_path), conf=conf, iou=iou)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"could not open {video}")

    rows = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        boxes, scores, ms = det(frame)
        mean_conf = float(scores.mean()) if len(scores) else 0.0
        rows.append((idx, len(boxes), round(mean_conf, 4), round(ms, 3)))
        idx += 1
    cap.release()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "num_detections", "mean_confidence", "latency_ms"])
        w.writerows(rows)

    latencies = sorted(r[3] for r in rows)
    total_dets = sum(r[1] for r in rows)
    frames_with_dets = sum(1 for r in rows if r[1] > 0)
    mean_conf = (
        sum(r[2] for r in rows if r[1] > 0) / frames_with_dets if frames_with_dets else 0.0
    )
    return {
        "frames": len(rows),
        "total_detections": total_dets,
        "frames_with_dets": frames_with_dets,
        "mean_confidence": round(mean_conf, 4),
        "mean_latency_ms": round(sum(latencies) / max(len(latencies), 1), 3),
        "p95_latency_ms": round(latencies[int(0.95 * len(latencies))], 3) if latencies else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="runs/detect/defect_v1/weights")
    ap.add_argument("--video", default="test_video.mp4")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out-dir", default="runs/bench")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    video = Path(args.video)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = [
        ("fp32", weights_dir / "best.onnx"),
        ("fp16", weights_dir / "best_fp16.onnx"),
        ("int8", weights_dir / "best_int8.onnx"),
    ]

    results = {}
    for mode, model_path in variants:
        if not model_path.exists():
            print(f"[skip {mode}] {model_path} not found")
            continue
        size_mb = round(model_path.stat().st_size / 1e6, 2)
        print(f"[bench {mode}]  {model_path.name}  ({size_mb} MB)")
        r = bench_one(model_path, video, args.conf, args.iou, out_dir / f"{mode}.csv")
        r["size_mb"] = size_mb
        results[mode] = r

    if not results:
        raise SystemExit("no models found to benchmark")

    header = ["mode", "size_mb", "mean_ms", "p95_ms", "total_det", "frames_det", "mean_conf"]
    widths = [6, 8, 8, 8, 10, 11, 10]
    print("\n" + "  ".join(h.rjust(w) for h, w in zip(header, widths)))
    for mode, r in results.items():
        row = [mode, r["size_mb"], r["mean_latency_ms"], r["p95_latency_ms"],
               r["total_detections"], r["frames_with_dets"], r["mean_confidence"]]
        print("  ".join(str(v).rjust(w) for v, w in zip(row, widths)))

    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["frames"])
        for mode, r in results.items():
            w.writerow([mode, r["size_mb"], r["mean_latency_ms"], r["p95_latency_ms"],
                        r["total_detections"], r["frames_with_dets"], r["mean_confidence"], r["frames"]])
    print(f"\nsummary -> {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
