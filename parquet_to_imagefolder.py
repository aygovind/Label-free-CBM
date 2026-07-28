#!/usr/bin/env python
"""
Convert the HuggingFace parquet version of BIRDS 525 SPECIES into the
ImageFolder layout that Label-free-CBM expects:

    data/birds525/train/<CLASS NAME>/000123.jpg
    data/birds525/valid/<CLASS NAME>/...
    data/birds525/test/<CLASS NAME>/...

Writes the stored JPEG bytes straight to disk -- no decode/re-encode, so
there is no generation loss and it runs at disk speed.

Usage:
    python parquet_to_imagefolder.py \
        --src data/birds525 \
        --dst data/birds525_if
"""
import argparse
import io
import os
import sys
from collections import Counter

from datasets import load_dataset
from datasets import Image as HFImage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dir containing the parquet shards")
    ap.add_argument("--dst", required=True, help="output ImageFolder root")
    ap.add_argument("--image-col", default=None, help="override image column name")
    ap.add_argument("--label-col", default=None, help="override label column name")
    args = ap.parse_args()

    print(f"loading parquet from {args.src} ...")
    ds = load_dataset(args.src)
    print(ds)

    first = ds[list(ds.keys())[0]]
    cols = first.column_names

    img_col = args.image_col or next(
        (c for c in ("image", "img", "jpg", "picture") if c in cols), None)
    lbl_col = args.label_col or next(
        (c for c in ("label", "labels", "class", "species") if c in cols), None)

    if img_col is None or lbl_col is None:
        sys.exit(f"could not identify columns; found {cols}. "
                 f"Pass --image-col / --label-col explicitly.")
    print(f"image column: {img_col}   label column: {lbl_col}")

    # Recover the 525 species names from the ClassLabel feature.
    feat = first.features[lbl_col]
    names = getattr(feat, "names", None)
    if names:
        print(f"found {len(names)} class names, e.g. {names[:3]}")
    else:
        n = len(set(first[lbl_col]))
        names = [f"class_{i:04d}" for i in range(n)]
        print(f"WARNING: no ClassLabel names; falling back to {n} numeric names. "
              f"Class identities will be meaningless -- prefer the Kaggle download.")

    # Sanity: HF's imagefolder builder sorts class names, and torchvision's
    # ImageFolder sorts directory names, so the two orderings should agree.
    if names != sorted(names):
        print("NOTE: ClassLabel order is not sorted. Directory names are still "
              "written verbatim and birds525_classes.txt is generated from the "
              "sorted directory listing, so label indices stay consistent.")

    total = 0
    for split in ds.keys():
        d = ds[split].cast_column(img_col, HFImage(decode=False))
        out_root = os.path.join(args.dst, split)
        counts = Counter()

        print(f"\nwriting split '{split}' ({len(d)} images) -> {out_root}")
        for i, ex in enumerate(d):
            cls = names[ex[lbl_col]]
            cls_dir = os.path.join(out_root, cls)
            os.makedirs(cls_dir, exist_ok=True)

            rec = ex[img_col]
            data = rec.get("bytes")
            if data is None:                      # bytes not inlined
                with open(rec["path"], "rb") as fh:
                    data = fh.read()

            counts[cls] += 1
            with open(os.path.join(cls_dir, f"{counts[cls]:05d}.jpg"), "wb") as fh:
                fh.write(data)

            if (i + 1) % 5000 == 0:
                print(f"  {i + 1}/{len(d)}")

        total += len(d)
        print(f"  done: {len(d)} images across {len(counts)} classes")

    print(f"\nwrote {total} images to {args.dst}")


if __name__ == "__main__":
    main()
