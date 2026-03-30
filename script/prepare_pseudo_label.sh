#!/bin/bash


# Check if the user provided an argument
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_path> <output_path> <scale> [max_num_objects]"
    echo "  input_path:      directory containing images/ and sparse/ (COLMAP data)."
    echo "                   This folder is never modified."
    echo "  output_path:     directory where all generated files will be written."
    echo "  scale:           downscale factor for DEVA (use 1 for no downscaling)."
    echo "  max_num_objects: maximum number of objects DEVA can track (default: 100)."
    echo "                   Must match num_classes - 1 in your training config JSON."
    exit 1
fi


# Resolve absolute paths so they stay valid after cd into subdirectories.
input_path="$(realpath "$1")"
output_path="$(realpath -m "$2")"   # -m: OK if path does not exist yet
scale="$3"
max_num_objects="${4:-100}"

if [ ! -d "$input_path" ]; then
    echo "Error: input_path '$input_path' does not exist."
    exit 2
fi

mkdir -p "$output_path"


# 0. Normalise image resolutions (always) + optionally downscale by factor.
#    resize_images.py runs a resolution-consistency check as its first step
#    regardless of the scale factor, so it is called unconditionally.
#    Images that already share a single resolution are left untouched.
#    Output is always written to <output_path>/images_<scale>/ so that
#    the input directory is never modified.
echo "==> Normalising image resolutions + resizing by factor ${scale}x (if > 1) ..."
python script/resize_images.py \
    --image_dir  "${input_path}/images" \
    --factor     "$scale" \
    --output_dir "${output_path}/images_${scale}"

# When scale <= 1 no downscaled copy is produced; use the (normalised)
# source images directory directly.  Otherwise use the resized output.
if [ "$scale" -le 1 ]; then
    deva_img_path="${input_path}/images"
else
    deva_img_path="${output_path}/images_${scale}"
fi


# 1. DEVA anything mask
#    Images are read from $deva_img_path (set above).
#    Masks are written directly to <output_path>/Annotations/ — no copy needed.
cd Tracking-Anything-with-DEVA/

# colored mask for visualization check (disabled by default)
#python demo/demo_automatic.py \
#  --chunk_size 4 \
#  --img_path "${output_path}/images_${scale}" \
#  --amp \
#  --temporal_setting semionline \
#  --size 480 \
#  --output "$output_path" \
#  --suppress_small_objects  \
#  --SAM_PRED_IOU_THRESHOLD 0.7 \
#  --max_num_objects "$max_num_objects" \

# gray mask for training
python demo/demo_automatic.py \
  --chunk_size 4 \
  --img_path   "${deva_img_path}" \
  --amp \
  --temporal_setting semionline \
  --size 768 \
  --output     "$output_path" \
  --use_short_id  \
  --suppress_small_objects  \
  --SAM_PRED_IOU_THRESHOLD 0.7 \
  --max_num_objects "$max_num_objects"

cd ..


# 2. CLIP post-hoc classification: overwrite generic labels with predicted class names.
#    Runs once on all unique objects (not per-frame) — typically adds < 1 s on GPU.
echo "==> Running CLIP post-hoc object classification ..."
python script/label_objects_clip.py \
    --pred_json  "${output_path}/pred.json" \
    --mask_dir   "${output_path}/Annotations" \
    --image_dir  "${deva_img_path}" \
    --output     "${output_path}/id2label.json"