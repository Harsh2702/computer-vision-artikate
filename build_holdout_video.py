"""
Build a held-out inference video from the training split, using stratified sampling
so both the kept training set and the video contain a mix of defect and background frames.

Defect vs background is decided by label file content: a label file with at least one
box line is a defect image; empty label file is a background image.

Usage:
    python build_holdout_video.py \
        --root defect_data \
        --keep-defect 22 \
        --keep-bg 278 \
        --fps 20 \
        --out test_video.mp4

The remaining images (defect + background) go into the video and are then deleted
from the training split along with their labels.
"""

import argparse
import random
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def load_pairs(train_root: Path):
    img_dir = train_root / "images"
    lbl_dir = train_root / "labels"
    defect, background = [], []
    for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS):
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0 and any(
            line.strip() for line in lbl.read_text().splitlines()
        ):
            defect.append(img)
        else:
            background.append(img)
    return defect, background


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--keep-defect", type=int, default=22)
    ap.add_argument("--keep-bg", type=int, default=278)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", default="test_video.mp4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train = Path(args.root) / "train"
    lbl_dir = train / "labels"

    defect, background = load_pairs(train)
    print(f"train pool: {len(defect)} defect, {len(background)} background")
    if len(defect) < args.keep_defect:
        raise SystemExit(f"only {len(defect)} defect images, cannot keep {args.keep_defect}")
    if len(background) < args.keep_bg:
        raise SystemExit(f"only {len(background)} bg images, cannot keep {args.keep_bg}")

    rng = random.Random(args.seed)
    rng.shuffle(defect)
    rng.shuffle(background)
    keep_defect = defect[: args.keep_defect]
    keep_bg = background[: args.keep_bg]
    vid_defect = defect[args.keep_defect :]
    vid_bg = background[args.keep_bg :]

    print(f"keeping in train: {len(keep_defect)} defect + {len(keep_bg)} bg = {len(keep_defect) + len(keep_bg)}")
    print(f"video frames:     {len(vid_defect)} defect + {len(vid_bg)} bg = {len(vid_defect) + len(vid_bg)}")

    video_frames = vid_defect + vid_bg
    rng.shuffle(video_frames)
    print(f"video duration:   {len(video_frames) / args.fps:.1f}s @ {args.fps} fps")

    first = cv2.imread(str(video_frames[0]))
    if first is None:
        raise SystemExit(f"failed to read {video_frames[0]}")
    h, w = first.shape[:2]
    print(f"frame size:       {w}x{h}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"cv2.VideoWriter failed to open {args.out}")

    written = 0
    for img_path in video_frames:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  skip unreadable: {img_path.name}")
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        writer.write(frame)
        written += 1
    writer.release()
    print(f"wrote {written} frames -> {args.out}")

    for img_path in video_frames:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img_path.unlink(missing_ok=True)
        if lbl_path.exists():
            lbl_path.unlink()

    remaining_imgs = len(list((train / "images").iterdir()))
    remaining_lbls = len(list(lbl_dir.iterdir()))
    non_empty = sum(1 for p in lbl_dir.iterdir() if p.stat().st_size > 0)
    print(f"train/ now has: {remaining_imgs} images, {remaining_lbls} labels ({non_empty} non-empty)")


if __name__ == "__main__":
    main()
