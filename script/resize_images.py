#!/usr/bin/env python3
"""
Resize an image folder by a given integer downscale factor.
Modelled after gsplat/examples/datasets/colmap.py::_resize_image_folder.

Usage:
    python script/resize_images.py --image_dir data/<dataset>/images --factor 2
The resized images are written to  data/<dataset>/images_<factor>/

NEW (normalisation step):
    Before downscaling, the script checks whether all images in --image_dir share
    the same resolution.  If they do not, it renames the original folder to
    <image_dir>_org, resizes every image to the most-common resolution, and saves
    the results back into <image_dir>.  This makes the folder safe for DEVA
    tracking, which requires a consistent spatial size across all frames.
"""

import argparse
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import List

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

# Pillow ≥ 9 deprecates the top-level resampling constants.
_BICUBIC = getattr(Image, "Resampling", Image).BICUBIC

# ---------------------------------------------------------------------------
# Normalisation threshold
# If the fraction of images that already share the most-common resolution is
# >= this value the normalisation step is skipped entirely.
# 1.0  → skip only when ALL images are identical in size  (recommended)
# 0.9  → skip when 90 % or more images share the same size
# ---------------------------------------------------------------------------
_UNIFORM_THRESHOLD = 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_rel_paths(path_dir: str) -> List[str]:
    """Recursively collect relative paths of every file inside *path_dir*."""
    paths = []
    for dp, _dn, fn in os.walk(path_dir):
        for f in fn:
            paths.append(os.path.relpath(os.path.join(dp, f), path_dir))
    return sorted(paths)


def _is_image_file(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ---------------------------------------------------------------------------
# NEW: resolution-normalisation step
# ---------------------------------------------------------------------------

def normalize_image_folder(image_dir: str) -> str:
    """
    Scan every image in *image_dir* and check whether they all share the same
    resolution.

    * If they do (fraction of most-common size >= _UNIFORM_THRESHOLD) → no-op.
    * Otherwise:
        1. Rename  ``<image_dir>``      →  ``<image_dir>_org``  (original backup)
        2. Create  ``<image_dir>``      (fresh folder)
        3. Copy images that already have the target size unchanged; resize the
           rest to the most-common (target) size using bicubic interpolation.

    Re-running after a previous normalisation is safe: the presence of
    ``<image_dir>_org`` is used as a sentinel to skip the step.

    Returns the (possibly unchanged) *image_dir* path for use by subsequent
    steps.
    """
    image_dir = image_dir.rstrip("/\\")
    org_dir   = image_dir + "_org"

    # ------------------------------------------------------------------ #
    # Guard: if backup already exists, normalisation was done previously. #
    # ------------------------------------------------------------------ #
    if os.path.isdir(org_dir):
        print(
            f"[normalize] '{org_dir}' already exists – "
            "normalisation was already applied. Skipping."
        )
        return image_dir

    image_files = [f for f in _get_rel_paths(image_dir) if _is_image_file(f)]
    if not image_files:
        print(f"[normalize] No images found in '{image_dir}'. Skipping normalisation.")
        return image_dir

    # ------------------------------------------------------------------ #
    # Count resolutions                                                    #
    # ------------------------------------------------------------------ #
    size_counter: Counter = Counter()
    for rel in tqdm(image_files, desc="[normalize] scanning sizes"):
        with Image.open(os.path.join(image_dir, rel)) as img:
            size_counter[img.size] += 1   # img.size = (width, height)

    most_common_size, most_common_count = size_counter.most_common(1)[0]
    total    = len(image_files)
    fraction = most_common_count / total

    print(f"[normalize] {len(size_counter)} distinct resolution(s) across {total} image(s):")
    for size, count in size_counter.most_common():
        marker = "  ← target" if size == most_common_size else ""
        print(f"             {size[0]}×{size[1]}: {count} image(s){marker}")

    # ------------------------------------------------------------------ #
    # Decision                                                             #
    # ------------------------------------------------------------------ #
    if fraction >= _UNIFORM_THRESHOLD:
        print(
            f"[normalize] {fraction:.1%} of images already share "
            f"{most_common_size[0]}×{most_common_size[1]} – no normalisation needed."
        )
        return image_dir

    # ------------------------------------------------------------------ #
    # Normalise                                                            #
    # ------------------------------------------------------------------ #
    target_w, target_h = most_common_size
    print(
        f"[normalize] Only {fraction:.1%} share the most-common size "
        f"({target_w}×{target_h}). Normalising all {total} image(s) to that size …"
    )
    print(f"[normalize] Renaming '{image_dir}'  →  '{org_dir}' (original backup) …")
    os.rename(image_dir, org_dir)
    os.makedirs(image_dir, exist_ok=True)

    for rel in tqdm(image_files, desc="[normalize] normalising"):
        src = os.path.join(org_dir,    rel)
        dst = os.path.join(image_dir,  rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)

        with Image.open(src) as img:
            if img.size == most_common_size:
                # Already the right size – copy without re-encoding to avoid quality loss.
                shutil.copy2(src, dst)
            else:
                img.resize((target_w, target_h), _BICUBIC).save(dst)

    print(
        f"[normalize] Done. All {total} image(s) saved at "
        f"{target_w}×{target_h} in '{image_dir}'."
    )
    return image_dir


# ---------------------------------------------------------------------------
# core resize logic
# ---------------------------------------------------------------------------

def resize_image_folder(image_dir: str, resized_dir: str, factor: int) -> str:
    """
    Downscale every image in *image_dir* by *factor* and save the results in
    *resized_dir*.  Individual images that already exist in *resized_dir* are
    skipped, so the function is safe to call incrementally.

    Returns
    -------
    resized_dir : str
        Path to the directory that contains the resized images.
    """
    image_files = [f for f in _get_rel_paths(image_dir) if _is_image_file(f)]

    if not image_files:
        print(f"[resize_images] No image files found in '{image_dir}'. Nothing to do.")
        return resized_dir

    # ------------------------------------------------------------------
    # Early-exit check: if the output folder already exists and already
    # contains the same number of images we assume it is complete.
    # ------------------------------------------------------------------
    if os.path.isdir(resized_dir):
        existing = [f for f in _get_rel_paths(resized_dir) if _is_image_file(f)]
        if len(existing) >= len(image_files):
            print(
                f"[resize_images] '{resized_dir}' already exists with "
                f"{len(existing)} image(s) (source has {len(image_files)}). "
                "Skipping resize."
            )
            return resized_dir
        else:
            print(
                f"[resize_images] '{resized_dir}' exists but only has "
                f"{len(existing)}/{len(image_files)} image(s). "
                "Resuming resize…"
            )

    os.makedirs(resized_dir, exist_ok=True)
    print(
        f"[resize_images] Downscaling {len(image_files)} image(s) by {factor}x  "
        f"'{image_dir}' → '{resized_dir}'"
    )

    for image_file in tqdm(image_files, desc="resizing"):
        image_path = os.path.join(image_dir, image_file)
        resized_path = os.path.join(
            resized_dir, os.path.splitext(image_file)[0] + ".png"
        )
        # create any necessary sub-directories
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)

        if os.path.isfile(resized_path):
            continue  # already processed

        image = imageio.imread(image_path)
        if image.ndim == 2:
            # grayscale → keep as-is
            pass
        else:
            image = image[..., :3]  # drop alpha if present

        resized_size = (
            int(round(image.shape[1] / factor)),  # width
            int(round(image.shape[0] / factor)),  # height
        )
        resized_image = np.array(
            Image.fromarray(image).resize(resized_size, _BICUBIC)
        )
        imageio.imwrite(resized_path, resized_image)

    print(f"[resize_images] Done. Resized images saved to '{resized_dir}'.")
    return resized_dir


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resize an image folder by a downscale factor."
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        help="Path to the source images directory (e.g. data/garden/images).",
    )
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="Integer downscale factor (e.g. 2 → half resolution).",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Path to write resized images. "
            "Defaults to <image_dir>_<factor>  (e.g. data/garden/images_2)."
        ),
    )
    args = parser.parse_args()

    image_dir   = args.image_dir.rstrip("/")
    resized_dir = args.output_dir or f"{image_dir}_{args.factor}"

    # ------------------------------------------------------------------
    # Step 0 (NEW): always normalise to a single resolution first so that
    # downstream tools (e.g. DEVA tracking) never see mixed frame sizes.
    # ------------------------------------------------------------------
    normalize_image_folder(image_dir)

    # ------------------------------------------------------------------
    # Step 1: downscale by factor (unchanged logic).
    # If factor <= 1 there is nothing left to do – normalisation above is
    # the only job in that case.
    # ------------------------------------------------------------------
    if args.factor <= 1:
        # OLD behaviour (kept for reference):
        # print(f"[resize_images] factor={args.factor} ≤ 1 – nothing to resize.")
        # return
        print(f"[resize_images] factor={args.factor} ≤ 1 – normalisation done, no downscaling needed.")
        return

    resize_image_folder(image_dir, resized_dir, args.factor)


if __name__ == "__main__":
    main()



