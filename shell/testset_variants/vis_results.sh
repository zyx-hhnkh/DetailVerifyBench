cd "$(dirname "$0")/../.."

# path to experiment output directory
# path to test image directory
python visualization/visualize_results.py \
    --exp_dir PATH_TO_EXP_DIR \
    --image_root_dir PATH_TO_IMAGE_DIR
