"""ONNX Runtime inference wrapper for the casting-defect YOLOv8 model.

Runs the exported ONNX model on an image, folder, or video file.
Prints per-frame detections and inference latency; optionally logs to CSV.

Usage:
    python infer.py --model runs/detect/defect_v1/weights/best.onnx --source test_video.mp4 --csv runs/bench/fp32.csv
    python infer.py --model runs/detect/defect_v1/weights/best.onnx --source defect_data/valid/images/foo.jpg
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


class Detector:
    def __init__(self, onnx_path, imgsz=640, conf=0.25, iou=0.45, providers=None):
        self.imgsz = imgsz
        self.conf_thres = conf
        self.iou_thres = iou
        providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_dtype = np.float16 if "float16" in inp.type else np.float32

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = min(self.imgsz / h, self.imgsz / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_top = (self.imgsz - new_h) // 2
        pad_bot = self.imgsz - new_h - pad_top
        pad_left = (self.imgsz - new_w) // 2
        pad_right = self.imgsz - new_w - pad_left
        padded = cv2.copyMakeBorder(
            resized, pad_top, pad_bot, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        return padded, scale, pad_left, pad_top

    def _preprocess(self, img):
        padded, scale, pad_left, pad_top = self._letterbox(img)
        x = padded.astype(np.float32) / 255.0
        x = x.transpose(2, 0, 1)[None]
        return x.astype(self.input_dtype), scale, pad_left, pad_top

    @staticmethod
    def _iou_matrix(a, b):
        x1 = np.maximum(a[:, 0:1], b[:, 0])
        y1 = np.maximum(a[:, 1:2], b[:, 1])
        x2 = np.minimum(a[:, 2:3], b[:, 2])
        y2 = np.minimum(a[:, 3:4], b[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        union = area_a[:, None] + area_b - inter
        return inter / np.maximum(union, 1e-9)

    def _nms(self, boxes, scores):
        idxs = np.argsort(scores)[::-1]
        keep = []
        while len(idxs):
            i = idxs[0]
            keep.append(i)
            if len(idxs) == 1:
                break
            ious = self._iou_matrix(boxes[i:i + 1], boxes[idxs[1:]])[0]
            idxs = idxs[1:][ious < self.iou_thres]
        return keep

    def __call__(self, img):
        """Run inference on a single image (H, W, 3, BGR).
        Returns (boxes_xyxy, scores, latency_ms). Coords are in original image space.
        """
        orig_h, orig_w = img.shape[:2]
        x, scale, pad_left, pad_top = self._preprocess(img)
        t0 = time.perf_counter()
        raw = self.session.run(None, {self.input_name: x})[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        # YOLOv8 ONNX output shape: (1, 4+nc, N). Transpose to (N, 4+nc).
        raw = raw[0].T  # (N, 4+nc)
        boxes_xywh = raw[:, :4]
        cls_scores = raw[:, 4:]
        conf = cls_scores.max(axis=1)

        mask = conf >= self.conf_thres
        boxes_xywh = boxes_xywh[mask]
        conf = conf[mask]
        if len(boxes_xywh) == 0:
            return np.zeros((0, 4)), np.zeros((0,)), latency_ms

        x_c, y_c, w, h = boxes_xywh.T
        boxes_xyxy = np.stack([x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2], axis=1).astype(np.float32)

        keep = self._nms(boxes_xyxy, conf)
        boxes_xyxy = boxes_xyxy[keep]
        conf = conf[keep]

        # Unscale back to original image space.
        boxes_xyxy[:, [0, 2]] -= pad_left
        boxes_xyxy[:, [1, 3]] -= pad_top
        boxes_xyxy /= scale
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
        return boxes_xyxy, conf, latency_ms


def _run_video(det, src, csv_path=None):
    cap = cv2.VideoCapture(str(src))
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
    if not rows:
        print("no frames read from source")
        return
    lat = [r[3] for r in rows]
    print(f"frames: {len(rows)}  "
          f"mean_latency_ms: {sum(lat) / len(lat):.2f}  "
          f"p95_latency_ms: {sorted(lat)[int(0.95 * len(lat))]:.2f}  "
          f"total_detections: {sum(r[1] for r in rows)}")
    if csv_path:
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "num_detections", "mean_confidence", "latency_ms"])
            w.writerows(rows)
        print(f"log -> {csv_path}")


def _run_image(det, src):
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"could not read {src}")
    boxes, scores, ms = det(img)
    print(f"{src}: detections={len(boxes)}  latency_ms={ms:.2f}")
    for b, s in zip(boxes, scores):
        print(f"  {s:.3f}  [{b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}, {b[3]:.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    det = Detector(args.model, imgsz=args.imgsz, conf=args.conf, iou=args.iou)
    src = Path(args.source)
    if src.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}:
        _run_video(det, src, args.csv)
    else:
        _run_image(det, src)


if __name__ == "__main__":
    main()
