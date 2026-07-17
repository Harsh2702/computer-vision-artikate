# Section 1 — Diagnose a Failing CV Pipeline

## Scenario A — Accuracy drops after quantization

**Check first, in order:**

1. **Run the ONNX FP32 model on the same val set.** If mAP holds at 0.91, export is clean. If it dropped, the export is broken and INT8 is a red herring.

2. **Run an FP16 export on the same val set.** This is the key isolation step. FP16 sits between FP32 and INT8, and the result tells me where the loss lives:
   - **FP16 ≈ 0.91:** export and preprocessing are clean. Loss is purely INT8. Look at calibration and sensitive layers.
   - **FP16 ≈ 0.58:** not a quantization problem at all. It's export or preprocessing. Chasing calibration wastes time.
   - **FP16 in between (say 0.75):** partial export/preprocessing issue plus INT8 damage on top. Fix both.

3. **Sanity check NMS and confidence thresholds on both sides.** Sometimes FP32 eval and the INT8 pipeline use different thresholds. The gap is just that.

4. **Dump input tensors going into FP32 vs INT8 for the same image, compare numerically.** If they differ before inference runs, it was never quantization. Usual culprits: BGR vs RGB, normalization, letterbox color, resize interpolation, 0-1 vs 0-255.

**Three independent root causes:**

- **Bad PTQ calibration set.** Activation ranges clip wrong when calibration doesn't cover the defect distribution. Test: recalibrate on 500 stratified images across all classes. mAP recovers → that was it.

- **Sensitive layers quantized.** Detection head and first conv are usual suspects. Test: mixed precision, head and first/last layers FP16, backbone INT8. mAP jumps back near 0.91 → found it.

- **Preprocessing mismatch on Jetson.** Test: the tensor dump from step 4. Inputs don't match numerically → model was never the problem.

**Fix and validation:** Usually mixed precision plus a proper calibration set. Before redeploy: run INT8 engine on full val set on the actual Orin (not the dev box), confirm mAP within 2-3 points of FP32, shadow-deploy 24-48 hours logging both models before cutover.

---

## Scenario B — Bounding boxes drift on one camera

**What the pattern tells me:** Systematic + edge-worse-than-center is the fingerprint of a geometric transform mismatch, not a model bug. If the model were wrong, errors would be random or class-dependent. Edge-dependence specifically points to either lens distortion that's being corrected on 11 feeds and not this one, or an aspect-ratio/resize mismatch where coordinates get mapped back to the wrong original frame size. The model is fine. The coordinate math around it is not.

**What I'd check on that one feed:**
- Resolution and aspect ratio of the RTSP stream vs what the other 11 send. If this camera is 1920x1080 and the pipeline assumes 1280x720, boxes scale wrong and the error grows toward the edges.
- Letterbox vs stretch resize. If preprocessing letterboxes but postprocessing assumes stretch (or vice versa), boxes are offset and the offset is worst at the edges.
- Camera-specific calibration or undistortion step. If 11 feeds have distortion coefficients applied and this one doesn't (or has the wrong ones), fisheye/barrel distortion shows up exactly this way.
- Codec or rotation metadata. Some RTSP streams carry rotation flags that OpenCV ignores but ffmpeg respects, so the frame comes in rotated or with a different sensor crop.

**Root cause hypothesis, no physical access:** Aspect-ratio mismatch between this camera's actual stream resolution and the resolution the postprocess assumes when scaling boxes back to original. Confirm by pulling one frame from that feed, logging its shape at ingestion, logging the shape the model sees, and logging what the box-rescaling code assumes. One of those three numbers won't match the others.

---

## Scenario C — Silent drift over three months

**Three plausible causes:**

- **Lighting drift.** Factory floors change. New overhead LEDs got installed, a skylight got dirty, seasonal daylight through a window shifted. The model was trained on one lighting distribution and is now seeing another. Evidence: pull image histograms from week 1 vs week 12 and compare mean brightness / color distribution. If the histograms shifted, that's your answer. Confirm by re-evaluating the model on recent images that were manually relabeled.

- **Physical setup drift the client didn't report.** Camera nudged by a cleaner, belt speed changed, a new product SKU got added to the line that looks similar to existing classes. Evidence: sample 50 images per week over the three months, look for changes in object size distribution, position, or new visual patterns. Confirm by asking the client specifically about SKU changes and maintenance logs, not "did anything change" (they always say no).

- **Label drift on the ground truth side.** If accuracy is being measured against an evolving human-labeled sample, the humans may have gotten stricter or the sampling changed. Evidence: check who's labeling recent data and whether the labeling guide changed. This one is embarrassing but common.

**Lightweight monitoring signal:** Log the model's average detection confidence and class-distribution histogram per hour, then run a rolling KL divergence or PSI (Population Stability Index) against a fixed baseline from week 1. Confidence drift and prediction-distribution drift both show up 2-4 weeks before accuracy visibly craters. Alert when PSI crosses 0.2. This needs zero ground truth labels, runs on the edge, and would have flagged this within a week or two.

---

# Section 3 — Find the Silent Bug

**File and line.** `infer.py`, inside `Detector._letterbox`, the `cv2.resize(...)` call used `interpolation=cv2.INTER_NEAREST`. The intended value is `cv2.INTER_LINEAR`, which is what the ultralytics training pipeline uses. The buggy version was introduced in commit `c951c69` as a "perf" tweak. Fixed in commit `259e887`.

**What it does wrong.** Every input frame is resized before it hits the model. Nearest-neighbour interpolation snaps each output pixel to the closest source pixel, so any smooth intensity change gets replaced by a stepped one. The tensor that reaches the model looks different from the tensor the model was trained on: same shape, same intensity range, subtly different pixel values, especially along edges. YOLOv8 heads are trained to be moderately robust to small perturbations, so most predictions still land in roughly the right place, they just shift.

**Why it looks plausible most of the time.** Three reasons stack:

1. **The pipeline never errors.** Shapes, dtypes, coordinate math are all fine. It runs at full speed and produces boxes.
2. **Most boxes still fire.** For high-contrast defects the model's confidence is well above threshold either way, so top-1 detections rarely move. Only near-threshold cases flip.
3. **This dataset is grayscale.** I checked and B = G = R exactly on every casting image (mean absolute channel diff = 0). That means several classical silent-preprocessing bugs I first considered (BGR vs RGB, wrong pad colour) are truly invisible on this data because their affected channels or regions are identical. Interpolation was the one preprocessing knob that still measurably shifts pixel values on grayscale square inputs.

Numerically, on 12 defective validation images at `conf=0.05`, the buggy version reported 6 total detections vs the ultralytics reference's 5. One image flipped from 0 detections to 1. Mean max-confidence per image shifted by about 0.008. A visual demo would show the same boxes in the same places, only slightly bumpier.

**The fix.**

```python
resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
```

That is it. One argument. It matches the interpolation used by ultralytics's own preprocessing so the ONNX wrapper feeds the model the same distribution it was trained on.

**Test that would have caught it (added in the fix commit).** `tests/test_infer.py::test_onnx_matches_ultralytics_reference`. It picks the first 12 defective validation images, runs both the ultralytics reference (`YOLO(best.pt)(path)`) and our `Detector` on `best.onnx` at `conf=0.05`, and asserts two things: total detection count divergence per image is under 0.05, and the mean per-image max-confidence gap is under 0.005. On the buggy commit the test fails with `detection count diverges: ref=5, ours=6, gap/img=0.08 > 0.05`. After the fix it passes cleanly.

The threshold matters: I run the test at `conf=0.05` rather than a normal `0.25` on purpose, because on this weakly-trained model that is the region where near-threshold flips are actually observable. A stricter model would let me raise the threshold; the point is to test in the band where drift matters, not where the model is over-confident anyway.

**What the bug's existence says about the testing gap.** The original workflow had no test comparing our ONNX inference against the framework we trained with. Visual smoke checks passed, and every downstream step (export, video walkthrough) also passed because the pipeline doesn't crash. The gap is a missing behavioural equivalence test between the training-time inference path and the deployed inference path: every preprocessing knob (resize interpolation, channel order, normalisation, letterbox pad colour, dtype) has to be pinned by a test that runs both stacks on the same image and compares outputs within a tight tolerance. Adding this test once, and running it in CI on every commit that touches `infer.py`, would have caught this change the moment it was made.

---

# Section 4 — Edge & Air-Gapped Deployment Design

Numbers below fall into two buckets. **Measured on my box** (RTX 3050 Laptop, 4 GB, ONNX Runtime CPU EP because cuDNN 9 is not installed for the GPU EP): FP32 ONNX YOLOv8n at 640×640 runs at 30 ms/frame mean, 32 ms p95 on the held-out video. FP16 was slower (55 ms) because CPU has no native FP16 fast path. INT8 was 59 ms for the same reason plus dequantise overhead. Model sizes: FP32 12.3 MB, FP16 6.2 MB, INT8 3.4 MB. **Estimated / not measured here**: everything Orin-related, everything about 8-stream DeepStream, INT8 speedups on hardware that actually accelerates it. I flag those inline.

### 1. Model family and precision

**YOLOv8s at INT8**, FP16 fallback if INT8 costs too much mAP on the client's defects.

- YOLOv8n: too weak for small industrial defects
- YOLOv8m: eats too much latency budget across 8 streams
- INT8 over FP16: I need the headroom for 8 parallel feeds

Caveat: on Jetson Nano at Teksun, INT8 accuracy drop varied a lot by dataset. I'd run INT8 vs FP16 mAP on the client's actual defect data before committing. If INT8 drops a critical defect class below tolerance, FP16 is worth the latency hit.

### 2. Throughput arithmetic

**Load:** 8 × 15 fps = 120 fps aggregate.

**Budget:** 200ms end-to-end covers decode, preprocess, inference, postprocess. Inference itself needs to stay under 60-80ms per frame.

**Fit:** Published YOLOv8s INT8 benchmarks on Orin AGX sit in the 200-400 fps range for 640x640 single-stream. 120 fps aggregate should fit with margin. I haven't benchmarked Orin myself, so I'd verify before quoting the client.

**Architecture:** DeepStream pipeline. 8 RTSP feeds decoded on NVDEC, batched into one TensorRT engine at batch 8, demuxed for per-stream postprocess. Batching is the trick that makes 120 fps fit on one Orin.

**What I'd validate first:**
- INT8 mAP on client's defect data
- End-to-end latency with full pipeline, not inference alone
- NVDEC capacity for 8× 1080p15
- Thermal behavior under sustained load, factory floors are not cool rooms

### 3. Air-gapped retraining loop

**Feedback capture:** Operator UI on the on-prem server. Shows recent detections, operators flag false positives and false negatives. Each flag saves frame, model output, correction, timestamp to a local queue.

**Retraining:** Separate on-prem training box (RTX 6000-class GPU) on the same air-gapped LAN. Weekly, an on-site ML engineer pulls flagged samples, reviews label quality, merges into the training set, retrains. Model artifacts move over LAN, not USB.

**Validation before replacing prod:**
1. Growing held-out test set. New model must beat old on the full set, not just new samples.
2. Shadow deployment 48-72 hours. New model runs in parallel, logs outputs, doesn't act.
3. Only after clean shadow does it get promoted.

### 4. Rollback and regression detection

**Rollback:** Previous model engine stays on disk. Deploy via symlink swap or config version bump, never overwrite. Rollback = flip the symlink, restart DeepStream. Under 5 minutes, decision to running.

**Detection, two signals running always:**

1. **Prediction-distribution monitoring.** PSI on confidence and per-class detection rate against the old model's baseline. Sudden shifts on day 1 are a red flag without needing ground truth.
2. **Operator flag rate.** False-positive and false-negative flags per shift. Alert when it jumps 2-3 std above historical baseline.

Both run on the Orin, no network. Detection window: 4-24 hours for a bad regression, faster if the model collapses entirely.
