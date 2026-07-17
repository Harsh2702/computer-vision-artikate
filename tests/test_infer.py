"""Regression test: ONNX inference must match the ultralytics reference within tolerance.

If preprocessing drifts from what ultralytics does at train time (channel order,
normalization, letterbox padding, etc.), predictions still fire but confidences
and coordinates shift systematically. This test compares our Detector against
ultralytics YOLO on defective validation images and fails when the mean
confidence gap exceeds a small tolerance.

Run:
    pytest tests/test_infer.py -v
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from infer import Detector  # noqa: E402
from ultralytics import YOLO  # noqa: E402

WEIGHTS_PT = REPO / "runs" / "detect" / "defect_v1" / "weights" / "best.pt"
WEIGHTS_ONNX = REPO / "runs" / "detect" / "defect_v1" / "weights" / "best.onnx"
VAL_IMG_DIR = REPO / "defect_data" / "valid" / "images"
VAL_LBL_DIR = REPO / "defect_data" / "valid" / "labels"

CONF = 0.25
IOU = 0.45
N_IMAGES = 8
MEAN_CONF_TOL = 0.05
COUNT_DIFF_TOL_PER_IMG = 1.0


def _defect_images(n):
    picks = []
    for lbl in sorted(VAL_LBL_DIR.iterdir()):
        if lbl.suffix != ".txt":
            continue
        try:
            content = lbl.read_text()
        except OSError:
            continue
        if any(line.strip() for line in content.splitlines()):
            img_candidates = list(VAL_IMG_DIR.glob(lbl.stem + ".*"))
            if img_candidates:
                picks.append(img_candidates[0])
        if len(picks) >= n:
            break
    return picks


@pytest.mark.skipif(not WEIGHTS_ONNX.exists() or not WEIGHTS_PT.exists(),
                    reason="model weights not present, run training + export first")
def test_onnx_matches_ultralytics_reference():
    images = _defect_images(N_IMAGES)
    assert images, "no defective validation images with labels found"

    ref = YOLO(str(WEIGHTS_PT))
    det = Detector(str(WEIGHTS_ONNX), conf=CONF, iou=IOU)

    total_ref = 0
    total_ours = 0
    conf_gaps = []
    for img_path in images:
        r = ref(str(img_path), verbose=False, conf=CONF, iou=IOU)[0]
        ref_conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros((0,))
        img = cv2.imread(str(img_path))
        our_boxes, our_conf, _ = det(img)

        total_ref += len(ref_conf)
        total_ours += len(our_conf)
        if len(ref_conf) and len(our_conf):
            conf_gaps.append(abs(float(ref_conf.mean()) - float(our_conf.mean())))

    count_gap_per_img = abs(total_ref - total_ours) / len(images)
    assert count_gap_per_img <= COUNT_DIFF_TOL_PER_IMG, (
        f"detection count diverges: ref={total_ref}, ours={total_ours}, "
        f"gap/img={count_gap_per_img:.2f} > {COUNT_DIFF_TOL_PER_IMG}"
    )

    assert conf_gaps, "no overlapping detections between ref and ours, cannot compare"
    mean_gap = float(np.mean(conf_gaps))
    assert mean_gap <= MEAN_CONF_TOL, (
        f"mean confidence gap {mean_gap:.3f} exceeds tolerance {MEAN_CONF_TOL}. "
        f"Likely preprocessing mismatch (channel order, normalization, letterbox)."
    )
