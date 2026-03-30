#!/usr/bin/env python3
"""Post-hoc CLIP classification of segmented objects.

Reads pred.json (produced by DEVA) + mask PNGs + original images,
classifies each unique tracked object with CLIP by accumulating similarity
scores across ALL valid frames, and writes id2label.json.

Frames where an object appears only as a small clipped region at the image
boundary are excluded; the script falls back to the largest-area frame when
every appearance of an object would otherwise be excluded.

Memory-efficient design
-----------------------
Images are loaded one frame at a time (frame-first iteration) and freed
immediately after all objects in that frame have been processed.  CLIP logit
scores are accumulated incrementally so no large crop buffers are kept in RAM.
Use --load_max_dim (default 1024) to cap the loaded resolution — CLIP only
needs 224 × 224 crops, so full-resolution loading is never necessary.

Usage (called automatically by prepare_pseudo_label.sh):
    python script/label_objects_clip.py \
        --pred_json  <output>/pred.json \
        --mask_dir   <output>/Annotations \
        --image_dir  data/<name>/images \
        --output     <output>/id2label.json
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import clip
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Vocabulary: COCO 80 + common scene / architectural / outdoor / indoor terms.
# CLIP zero-shot works best with concrete nouns.  Edit freely.
# ---------------------------------------------------------------------------
VOCAB = [
    # COCO 80
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
    # Scene / architectural / outdoor / indoor
    'wall', 'floor', 'ceiling', 'sky', 'grass', 'tree', 'road', 'building',
    'window', 'door', 'fence', 'sidewalk', 'pavement', 'mountain', 'rock',
    'water', 'river', 'sea', 'sand', 'bush', 'flower', 'sign', 'pole',
    'column', 'stairs', 'railing', 'cabinet', 'shelf', 'table', 'lamp',
    'curtain', 'pillow', 'rug', 'mirror', 'painting', 'plant', 'sculpture',
    'streetlight', 'mailbox', 'trash can', 'canopy', 'bridge', 'tunnel',
    'wheel', 'tire', 'engine', 'pipe', 'wire', 'cable', 'antenna',
]
# deduplicate preserving order
VOCAB = list(dict.fromkeys(VOCAB))

# ---------------------------------------------------------------------------
# Prompt ensemble — averaging over multiple templates improves zero-shot
# accuracy and makes CLIP robust to partial / cropped views of objects.
# ---------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    'a photo of a {}',
    'a cropped photo of a {}',
    'a partial view of a {}',
    'a close-up photo of a {}',
    'a photo of part of a {}',
    'a photo of the {} in the scene',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def best_crop(image_np: np.ndarray, mask: np.ndarray, pad: int = 8) -> Image.Image:
    """Return the bounding-box crop of the masked region with a small padding."""
    ys, xs = np.where(mask)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, image_np.shape[0])
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, image_np.shape[1])
    return Image.fromarray(image_np[y0:y1, x0:x1])


def is_boundary_clipped(mask: np.ndarray, image_h: int, image_w: int,
                        border_frac: float = 0.02,
                        area_frac: float = 0.30,
                        max_area: int = 0) -> bool:
    """Return True if the object appears only as a clipped sliver at the border.

    A frame is considered boundary-clipped when BOTH conditions hold:
      (a) The mask bounding box touches the image edge within
          ``border_frac * min(H, W)`` pixels.
      (b) The mask area is below ``area_frac * max_area``, where
          ``max_area`` is the object's largest area across all frames.

    Condition (b) ensures that large scene elements (floor, sky, wall) that
    naturally span the full image are never excluded.
    """
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return True

    margin = max(2, int(border_frac * min(image_h, image_w)))
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())

    touches_border = (
        y0 <= margin or x0 <= margin
        or y1 >= image_h - 1 - margin
        or x1 >= image_w - 1 - margin
    )
    if not touches_border:
        return False

    # Keep large objects even when they touch the border
    area = int(mask.sum())
    if max_area > 0 and area >= area_frac * max_area:
        return False

    return True


def find_image(image_dir: str, stem: str) -> str | None:
    """Return the path of the image file matching *stem* in *image_dir*."""
    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        p = os.path.join(image_dir, stem + ext)
        if os.path.exists(p):
            return p
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_image(img_path: str, max_dim: int) -> np.ndarray:
    """Load an image as RGB numpy array, optionally downscaling the longest side."""
    img = Image.open(img_path).convert('RGB')
    if max_dim > 0:
        w, h = img.size
        longest = max(w, h)
        if longest > max_dim:
            scale = max_dim / longest
            img = img.resize((max(1, int(round(w * scale))),
                              max(1, int(round(h * scale)))),
                             Image.BILINEAR)
    return np.array(img)


def main():
    parser = argparse.ArgumentParser(
        description='Label segmented objects with CLIP zero-shot classification.')
    parser.add_argument('--pred_json',  required=True,
                        help='pred.json produced by demo_automatic.py')
    parser.add_argument('--mask_dir',   required=True,
                        help='Directory containing Annotations/*.png (gray masks)')
    parser.add_argument('--image_dir',  required=True,
                        help='Directory containing the original RGB images')
    parser.add_argument('--output',     required=True,
                        help='Output path for id2label.json')
    parser.add_argument('--clip_model', default='ViT-B/32',
                        help='CLIP model variant (default: ViT-B/32)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='CLIP batch size (default: 32)')
    parser.add_argument('--max_frames_per_object', type=int, default=0,
                        help='Cap frames per object (0 = no limit, default: 0)')
    parser.add_argument('--border_frac', type=float, default=0.02)
    parser.add_argument('--area_frac',   type=float, default=0.30)
    parser.add_argument('--load_max_dim', type=int, default=1024,
                        help='Cap the longest image dimension when loading to '
                             'save memory — images are downscaled proportionally '
                             'before crop extraction (0 = no cap, default: 1024). '
                             'CLIP only needs 224×224 crops so 1024 is plenty.')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── 1. Parse pred.json ────────────────────────────────────────────────────
    with open(args.pred_json) as f:
        pred = json.load(f)

    # obj_id → list of {'frame': str, 'area': int}
    obj_frames:   dict[int, list] = defaultdict(list)
    obj_max_area: dict[int, int]  = {}

    for ann in pred['annotations']:
        frame_name = ann['file_name']
        for seg in ann['segments_info']:
            obj_id = seg['id']
            area   = seg.get('area', 0)
            obj_frames[obj_id].append({'frame': frame_name, 'area': area})
            if area > obj_max_area.get(obj_id, 0):
                obj_max_area[obj_id] = area

    if not obj_frames:
        print('No objects found in pred.json — writing empty id2label.json')
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump({}, f, indent=4)
        return

    total_appearances = sum(len(v) for v in obj_frames.values())
    print(f'Found {len(obj_frames)} unique objects, '
          f'{total_appearances} total appearances across '
          f'{len(pred["annotations"])} frames')

    # ── 2. Build frame → objects index (with per-object subsampling) ──────────
    # frame_stem_key → set of obj_ids to evaluate when this frame is loaded.
    # frame_stem_key preserves subfolder prefix for multi-camera layouts
    # (e.g. 'pano_camera0/frame001') — matching the subfolder in mask_dir.
    frame_to_objs: dict[str, set] = defaultdict(set)
    for obj_id, appearances in obj_frames.items():
        if args.max_frames_per_object and len(appearances) > args.max_frames_per_object:
            step    = max(1, len(appearances) // args.max_frames_per_object)
            sampled = appearances[::step][:args.max_frames_per_object]
        else:
            sampled = appearances
        for entry in sampled:
            # str(Path(...).with_suffix('')) strips '.jpg'/'.png' but keeps
            # any subfolder prefix, e.g. 'pano_camera0/frame001'
            stem_key = str(Path(entry['frame']).with_suffix(''))
            frame_to_objs[stem_key].add(obj_id)

    # ── 3. Load CLIP + build prompt-ensemble text features ───────────────────
    print(f'Loading CLIP {args.clip_model} on {args.device} ...')
    model, preprocess = clip.load(args.clip_model, device=args.device)
    model.eval()

    print(f'Encoding {len(VOCAB)}-class vocabulary '
          f'with {len(PROMPT_TEMPLATES)}-template prompt ensemble ...')
    with torch.no_grad():
        template_feats = []
        for template in PROMPT_TEMPLATES:
            tokens = clip.tokenize(
                [template.format(c) for c in VOCAB]).to(args.device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            template_feats.append(feats)
        text_feats = torch.stack(template_feats).mean(dim=0)
        text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # ── 4. Frame-first crop extraction + incremental CLIP scoring ─────────────
    #
    # We iterate over frames exactly once.  For each frame we:
    #   (a) load the image at capped resolution and the mask — then free them
    #   (b) for every requested object in that frame: boundary-check, crop,
    #       run CLIP immediately and accumulate score vector
    #   (c) if all appearances of an object are boundary-clipped we keep the
    #       single best (largest-area) crop as a fallback (only one PIL image
    #       per such object is held in RAM at any time)
    #
    # This approach loads each image exactly once and never builds a large
    # in-memory crop buffer, making it feasible even for thousands of
    # high-resolution frames.

    obj_ids = sorted(obj_frames.keys())

    # Per-object CLIP score accumulator
    score_accum:    dict[int, torch.Tensor] = {
        oid: torch.zeros(len(VOCAB), device=args.device) for oid in obj_ids
    }
    obj_crop_count: dict[int, int] = {oid: 0 for oid in obj_ids}
    obj_skipped:    dict[int, int] = {oid: 0 for oid in obj_ids}
    # fallback_crop[obj_id] = (area, PIL.Image) — best boundary-clipped crop
    fallback_crop:  dict[int, tuple] = {}

    # Accumulate CLIP batches across objects within a frame for efficiency
    batch_crops: list = []
    batch_ids:   list = []

    def _flush_batch() -> None:
        if not batch_crops:
            return
        inp = torch.stack([preprocess(c) for c in batch_crops]).to(args.device)
        with torch.no_grad():
            feats  = model.encode_image(inp)
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            logits = feats @ text_feats.T          # [B, V]
        for j, oid in enumerate(batch_ids):
            score_accum[oid] += logits[j]
            obj_crop_count[oid] += 1
        batch_crops.clear()
        batch_ids.clear()

    all_frame_stems = sorted(frame_to_objs.keys())
    if args.load_max_dim > 0:
        print(f'Extracting crops frame-by-frame '
              f'(images capped at {args.load_max_dim}px, '
              f'{len(all_frame_stems)} frames to visit) ...')
    else:
        print(f'Extracting crops frame-by-frame '
              f'(full resolution, {len(all_frame_stems)} frames to visit) ...')

    for stem_key in tqdm(all_frame_stems, desc='Processing frames'):
        img_path  = find_image(args.image_dir, stem_key)
        mask_path = os.path.join(args.mask_dir, stem_key + '.png')

        if img_path is None or not os.path.exists(mask_path):
            continue

        image_np = _load_image(img_path, args.load_max_dim)
        mask_np  = np.array(Image.open(mask_path))

        # Align mask spatial size to (possibly downscaled) image
        if mask_np.shape[:2] != image_np.shape[:2]:
            mask_np = np.array(
                Image.fromarray(mask_np).resize(
                    (image_np.shape[1], image_np.shape[0]),
                    resample=Image.NEAREST))

        H, W = image_np.shape[:2]

        for obj_id in frame_to_objs[stem_key]:
            max_area = obj_max_area[obj_id]
            obj_mask = (mask_np == obj_id)
            if obj_mask.sum() == 0:
                continue

            area = int(obj_mask.sum())

            if is_boundary_clipped(obj_mask, H, W,
                                   border_frac=args.border_frac,
                                   area_frac=args.area_frac,
                                   max_area=max_area):
                obj_skipped[obj_id] += 1
                # Remember the single best boundary-clipped crop as last resort
                if obj_id not in fallback_crop or area > fallback_crop[obj_id][0]:
                    fallback_crop[obj_id] = (area, best_crop(image_np, obj_mask))
                continue

            batch_crops.append(best_crop(image_np, obj_mask))
            batch_ids.append(obj_id)
            if len(batch_crops) >= args.batch_size:
                _flush_batch()

        # Explicitly release the large arrays before the next frame
        del image_np, mask_np

    _flush_batch()

    # Report skipped / kept per object
    for oid in obj_ids:
        if obj_skipped[oid]:
            print(f'  Object {oid}: skipped {obj_skipped[oid]} boundary-clipped '
                  f'frames, kept {obj_crop_count[oid]}')

    # ── 5. Fallback: objects with zero valid crops ────────────────────────────
    # Use the best boundary-clipped crop we saved during the frame pass.
    fallback_ids = [oid for oid in obj_ids if obj_crop_count[oid] == 0]
    no_crop_ids  = []

    if fallback_ids:
        print(f'  {len(fallback_ids)} objects had all frames filtered; '
              f'using best boundary-clipped crop as fallback.')
        fb_crops, fb_ids = [], []
        for oid in fallback_ids:
            if oid in fallback_crop:
                fb_crops.append(fallback_crop[oid][1])
                fb_ids.append(oid)
            else:
                no_crop_ids.append(oid)

        for i in range(0, len(fb_crops), args.batch_size):
            inp = torch.stack(
                [preprocess(c) for c in fb_crops[i:i + args.batch_size]]
            ).to(args.device)
            b_ids = fb_ids[i:i + args.batch_size]
            with torch.no_grad():
                feats  = model.encode_image(inp)
                feats  = feats / feats.norm(dim=-1, keepdim=True)
                logits = feats @ text_feats.T
            for j, oid in enumerate(b_ids):
                score_accum[oid] += logits[j]
                obj_crop_count[oid] += 1

    if no_crop_ids:
        print(f'  Warning: {len(no_crop_ids)} objects have no usable crop '
              f'(will receive generic label): {no_crop_ids}')

    # ── 6. Build labels ───────────────────────────────────────────────────────
    id_to_label: dict[str, str] = {}
    for oid in obj_ids:
        if obj_crop_count[oid] > 0:
            best_idx = int(score_accum[oid].argmax().item())
            id_to_label[str(oid)] = VOCAB[best_idx]
        else:
            id_to_label[str(oid)] = f'object_{oid}'

    id_to_label = {k: id_to_label[k]
                   for k in sorted(id_to_label, key=lambda x: int(x))}

    # ── 7. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(id_to_label, f, indent=4)

    print(f'Saved → {args.output}')
    sample = dict(list(id_to_label.items())[:8])
    print(f'Sample labels: {sample}')


if __name__ == '__main__':
    main()

