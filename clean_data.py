"""
Validate a YOLO detection dataset and delete broken image/label pairs.

Rules for a pair to be kept:
  - image opens with PIL and reports valid dimensions
  - label file exists (empty is OK -> background)
  - every non-empty line has exactly 5 numeric fields
  - class index is in [0, nc-1]
  - cx, cy, w, h are in [0, 1]
  - w > 0 and h > 0

Any pair failing any check is deleted (both image and label file).
Duplicate label lines are deduped in place (kept, not deleted).

Usage:
    python clean_data.py --root defect_data --nc 1
    python clean_data.py --root defect_data --nc 1 --dry-run
"""

import argparse
from pathlib import Path
from PIL import Image, UnidentifiedImageError

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_label(path: Path, nc: int):
    """Return (ok, cleaned_lines, reason). cleaned_lines is deduped; reason set only if not ok."""
    try:
        raw = path.read_text().splitlines()
    except Exception as e:
        return False, [], f"unreadable ({e})"

    seen = set()
    cleaned = []
    for i, line in enumerate(raw, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            return False, [], f"line {i}: expected 5 fields, got {len(parts)}"
        try:
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            return False, [], f"line {i}: non-numeric field"
        if not 0 <= cls < nc:
            return False, [], f"line {i}: class {cls} out of range [0,{nc - 1}]"
        for name, v in ("cx", cx), ("cy", cy), ("w", w), ("h", h):
            if not 0.0 <= v <= 1.0:
                return False, [], f"line {i}: {name}={v} not in [0,1]"
        if w <= 0 or h <= 0:
            return False, [], f"line {i}: zero-area box"
        key = (cls, round(cx, 6), round(cy, 6), round(w, 6), round(h, 6))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(f"{cls} {cx} {cy} {w} {h}")
    return True, cleaned, ""


def validate_image(path: Path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
            if w <= 0 or h <= 0:
                return False, "zero dimensions"
    except (UnidentifiedImageError, OSError) as e:
        return False, f"unreadable ({e})"
    return True, ""


def clean_split(split_dir: Path, nc: int, dry_run: bool):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        print(f"[skip] {split_dir} missing images/ or labels/")
        return

    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
    labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"}

    orphan_imgs = set(images) - set(labels)
    orphan_lbls = set(labels) - set(images)
    paired = set(images) & set(labels)

    kept = 0
    deleted = 0
    background = 0
    boxes_kept = 0
    reasons = {}

    for stem in orphan_imgs:
        reasons.setdefault("orphan image (no label)", []).append(images[stem].name)
        if not dry_run:
            images[stem].unlink()
        deleted += 1
    for stem in orphan_lbls:
        reasons.setdefault("orphan label (no image)", []).append(labels[stem].name)
        if not dry_run:
            labels[stem].unlink()
        deleted += 1

    for stem in sorted(paired):
        img, lbl = images[stem], labels[stem]
        img_ok, img_reason = validate_image(img)
        if not img_ok:
            reasons.setdefault(f"image: {img_reason}", []).append(img.name)
            if not dry_run:
                img.unlink()
                lbl.unlink()
            deleted += 1
            continue

        ok, cleaned, reason = validate_label(lbl, nc)
        if not ok:
            reasons.setdefault(f"label: {reason}", []).append(lbl.name)
            if not dry_run:
                img.unlink()
                lbl.unlink()
            deleted += 1
            continue

        if not cleaned:
            background += 1
        else:
            boxes_kept += len(cleaned)
            if not dry_run:
                original = lbl.read_text().splitlines()
                deduped = [l for l in original if l.strip()]
                if len(deduped) != len(cleaned):
                    lbl.write_text("\n".join(cleaned) + "\n")
        kept += 1

    print(f"\n[{split_dir.name}]")
    print(f"  kept:       {kept}  ({background} background, {kept - background} with boxes)")
    print(f"  boxes kept: {boxes_kept}")
    print(f"  deleted:    {deleted}")
    if reasons:
        print("  reasons:")
        for r, files in reasons.items():
            preview = ", ".join(files[:3]) + (f", ... (+{len(files) - 3} more)" if len(files) > 3 else "")
            print(f"    - {r}: {len(files)}  [{preview}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root containing train/valid/test/")
    ap.add_argument("--nc", type=int, required=True, help="number of classes")
    ap.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    ap.add_argument("--dry-run", action="store_true", help="report only, do not delete")
    args = ap.parse_args()

    root = Path(args.root)
    print(f"root: {root.resolve()}")
    print(f"nc: {args.nc}")
    if args.dry_run:
        print("DRY RUN — no files will be deleted")

    for split in args.splits:
        clean_split(root / split, args.nc, args.dry_run)


if __name__ == "__main__":
    main()
