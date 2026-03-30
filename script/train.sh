#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_path> <output_path> <scale>"
    echo "  input_path:  directory containing images/ and sparse/ (COLMAP data, never modified)."
    echo "  output_path: directory produced by prepare_pseudo_label.sh (contains Annotations/)."
    echo "               Training artifacts are also written here."
    echo "  scale:       resolution divisor passed to the trainer (-r flag)."
    exit 1
fi

input_path="$(realpath "$1")"
output_path="$(realpath -m "$2")"
scale="$3"

if [ ! -d "$input_path" ]; then
    echo "Error: input_path '$input_path' does not exist."
    exit 2
fi

if [ ! -d "$output_path" ]; then
    echo "Error: output_path '$output_path' does not exist."
    echo "  Run prepare_pseudo_label.sh first to generate the segmentation masks."
    exit 3
fi


# Gaussian Grouping training.
# -s: COLMAP source (images + sparse/), read-only.
# -m: model output path (training artefacts written here).
# --object_path: absolute path to the gray masks produced by prepare_pseudo_label.sh.
python train.py \
    -s "$input_path" \
    -r "$scale" \
    -m "$output_path" \
    --object_path "${output_path}/Annotations" \
    --config_file config/gaussian_dataset/train.json

# Segmentation rendering using trained model
python render.py -m "$output_path" --num_classes 256

# Build segmented point cloud using trained model
python colorize_by_object_id.py -m "$output_path" --iteration 30000 --num_classes 256
