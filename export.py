"""Export trained YOLOv8 weights to ONNX at FP32, FP16, and INT8.

INT8 uses the training split for post-training calibration via ultralytics.

Usage:
    python export.py --weights runs/detect/defect_v1/weights/best.pt

Outputs (next to best.pt):
    best.onnx           # FP32
    best_fp16.onnx      # FP16
    best_int8.onnx      # INT8 (may be quantized in ONNXRuntime QDQ format)
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO, settings


def export_one(weights: Path, mode: str, data_yaml: Path, imgsz: int) -> Path:
    """Export at the given precision. Renames the produced ONNX to a mode-tagged name."""
    model = YOLO(str(weights))
    kwargs = {"format": "onnx", "imgsz": imgsz, "simplify": True, "opset": 12}
    if mode == "fp32":
        pass
    elif mode == "fp16":
        kwargs["half"] = True
    elif mode == "int8":
        kwargs["int8"] = True
        kwargs["data"] = str(data_yaml)
    else:
        raise ValueError(mode)

    produced = Path(model.export(**kwargs))
    print(f"  produced -> {produced}")

    target = weights.parent / f"best_{mode}.onnx" if mode != "fp32" else weights.parent / "best.onnx"
    if produced.resolve() != target.resolve():
        if target.exists():
            target.unlink()
        shutil.move(str(produced), str(target))
    print(f"  saved    -> {target}")
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="defect_data/data.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    # FP32 goes last: ultralytics writes fp32 and fp16 to the same "best.onnx" path,
    # so exporting FP32 first then FP16 silently overwrites the FP32 file.
    ap.add_argument("--modes", nargs="+", default=["fp16", "int8", "fp32"])
    args = ap.parse_args()

    repo = Path(__file__).parent.resolve()
    settings.update({"datasets_dir": str(repo)})

    weights = Path(args.weights).resolve()
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    data_yaml = Path(args.data).resolve()

    out = {}
    for mode in args.modes:
        print(f"[export {mode}]")
        out[mode] = export_one(weights, mode, data_yaml, args.imgsz)

    print("\nsummary:")
    for mode, p in out.items():
        print(f"  {mode:5s} {p.stat().st_size / 1e6:6.2f} MB  {p}")


if __name__ == "__main__":
    main()
