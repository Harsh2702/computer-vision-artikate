# computer-vision-artikate

Section 2 submission for the Artikate Studio Senior CV/ML Engineer take-home. Fine-tunes YOLOv8n on a small industrial-defect dataset, exports to ONNX at FP32/FP16/INT8, and benchmarks all three on a held-out video.

Written answers for Sections 1, 3, 4 live in [ANSWERS.md](ANSWERS.md).

## Hardware and env

- Windows 11, RTX 3050 Laptop (4 GB VRAM), Intel i5-12500H
- conda env `cuda112`, Python 3.8.20
- torch 2.4.1+cu121, ultralytics 8.4.96, onnxruntime-gpu 1.19.2

Training used the CUDA execution provider for PyTorch. ONNX Runtime benchmarks fell back to CPU because cuDNN 9 (required by ORT-GPU 1.19) is not installed in this env; the ORT CUDA fix is a straightforward `pip install nvidia-cudnn-cu12` on a machine with more time budget.

## Dataset

Source: [Roboflow Universe: casting-defect-adiic-rhmq6](https://universe.roboflow.com/tsd-9qn6z/casting-defect-adiic-rhmq6). 1,300 images of casting parts, one class (`defect`), YOLO-format labels, 512×512 grayscale. Chosen as the closest publicly available industrial-inspection stand-in for the private set the assignment describes. It gives real train / valid / test splits, real class imbalance (about 8% defective), and real-world label quality issues (one rogue class-1 label caught by `clean_data.py`).

After `clean_data.py` (drops broken labels, dedupes lines) and `build_holdout_video.py` (stratified: 22 defect + 278 background stay in `train/`; 50 defect + 558 background become `test_video.mp4` at 20 fps, 30.4 s):

| Split | Images | With defect boxes | Backgrounds |
|---|---|---|---|
| train | 300 | 22 | 278 |
| valid | 261 | 27 | 234 |
| test | 130 | 14 | 116 |
| held-out video | 608 frames | 50 defect | 558 background |

## What runs where

| File | Purpose |
|---|---|
| `clean_data.py` | Validate dataset, delete broken pairs, dedupe labels. Runs before every training run. |
| `build_holdout_video.py` | Stratified split of train → kept + held-out mp4. Seeded, reproducible. |
| `train_yolo.py` | Fine-tune YOLOv8n, 100 epochs w/ patience=20, batch=8, imgsz=640, seed=0. |
| `export.py` | Export best.pt → best.onnx, best_fp16.onnx, best_int8.onnx (INT8 uses ultralytics's calibration on the val split). |
| `infer.py` | ONNX Runtime inference wrapper (class `Detector`), letterbox + NMS + coord unscale. CLI supports image, folder, or video sources. |
| `bench.py` | Runs `infer.Detector` on the held-out video for each ONNX variant, logs per-frame CSV and prints a summary. |
| `failure_analysis.py` | Reads a benchmark CSV, picks top-confidence and borderline frames, dumps raw + annotated frames. |
| `tests/test_infer.py` | Regression test: `Detector` output must match ultralytics reference within tolerance. Catches the Section 3 bug. |

## Reproduce

```
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python clean_data.py --root defect_data --nc 1
python train_yolo.py                          # ~4 min on RTX 3050
python export.py --weights runs/detect/defect_v1/weights/best.pt
pytest tests/ -v
python bench.py --conf 0.05
python failure_analysis.py --csv runs/bench/fp32.csv --video test_video.mp4 --model runs/detect/defect_v1/weights/best.onnx --conf 0.05
```

## Training results

Best model saved at epoch 17, training early-stopped at epoch 37 by `patience=20`. Wall time 3.66 min.

On the validation split (261 images, 29 defect instances):

| Metric | Value |
|---|---|
| Precision | 0.227 |
| Recall | 0.547 |
| mAP@50 | 0.352 |
| mAP@50-95 | 0.117 |

Sanity: these numbers are honest for the training budget (22 defective train images is a small positive set; the assignment describes a similarly small private set). Recall is higher than precision, meaning the model over-predicts defects. The confusion is dominated by false positives on the background class.

Training also produced the standard evaluation plots in `runs/detect/defect_v1/`: `results.png` (losses and metrics per epoch), `confusion_matrix.png`, `BoxPR_curve.png`, `BoxF1_curve.png`, `labels.jpg`, and `val_batch{0,1,2}_{labels,pred}.jpg`.

## FP32 vs FP16 vs INT8 benchmark

Benchmark: `bench.py --conf 0.05` on `test_video.mp4` (608 frames), ORT CPU EP.

| Variant | Model size (MB) | Mean latency (ms) | p95 latency (ms) | Total detections | Frames with detections | Mean conf |
|---|---|---|---|---|---|---|
| FP32 | 12.27 | 30.28 | 32.10 | 67 | 55 | 0.066 |
| FP16 | 6.17 | 54.62 | 61.12 | 67 | 55 | 0.066 |
| INT8 | 3.39 | 58.84 | 66.08 | 66 | 55 | 0.064 |

Observations:

- **Size shrinks as expected**: FP16 is 50% of FP32, INT8 is 28%.
- **Latency does not track precision on CPU.** FP16 is nearly 2× slower than FP32 because ORT CPU EP has no native FP16 kernels for detection, so tensors get upcast to FP32 with overhead. INT8 is 2× slower than FP32 for the same reason plus per-op dequantise cost. This inversion is expected on CPU. On a GPU or NPU with real FP16/INT8 fast paths, the ordering flips.
- **Detection quality holds.** All three variants detect on the same 55 frames. INT8 lost one detection compared to FP32/FP16 and dropped mean confidence by 0.002, both consistent with static PTQ noise on a small model.
- Per-frame CSVs: `runs/bench/{fp32,fp16,int8}.csv`. Summary: `runs/bench/summary.csv`.

**Confidence that these numbers hold on a different machine.** Low, and honest about why: the CPU I ran on is a mobile i5-12500H, and ONNX Runtime CPU throughput is dominated by CPU generation, cache size, and thread count. On a server CPU expect maybe 2-3× faster. On the Orin AGX of Section 4, real INT8 through TensorRT would be an order of magnitude faster than any of these CPU numbers; those are not measured here.

## Worst-case frames from the held-out video

`failure_analysis.py` pulled the three highest-confidence detections (all around conf 0.09) and dumped raw and annotated frames to `runs/bench/failures/`. All three are false-positive-shaped: the model boxes a shadow or edge on an otherwise clean casting. Hypotheses:

1. **Frame 315.** Fires on the rim of the casting. Likely learned "curved dark edge" as a weak defect cue during training on the 22 positive examples.
2. **Frame 454.** Similar edge response, different casting orientation. Same failure mode.
3. **Frame 254.** Boxes a low-contrast region of the background. Suggests the background distribution in training (278 background frames) did not cover the exact lighting in this video frame.

None of these would survive a stricter confidence threshold, which is another way of saying the model's ceiling here is training-data-limited, not architecture-limited. With more positive examples and a matched background distribution the same YOLOv8n would push mAP well past 0.35.

## Git history

The commit log is the point. It walks: initial import, remove vendored ultralytics, gitignore data, clean_data, cleanup, add dataset + video, ONNX wrapper, export/bench/test scaffolding, **the buggy perf tweak**, **the fix + regression test**. That last pair is Section 3.
