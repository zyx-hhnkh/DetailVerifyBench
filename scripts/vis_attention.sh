PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python visualization/visualize_attention.py \
    --model_path PATH_TO_MODEL \
    --image_dir PATH_TO_IMAGE_DIR \
    --input_file PATH_TO_INFERENCE_RESULT_JSON \
    --output_dir PATH_TO_OUTPUT_DIR \
    --prompt_file prompt.json \
    --model_select Qwen3-VL-8B-Instruct \
    --max_samples 10 \
    --batch_size 30 \
    --use_think  # 如果推理时用了 CoT，这里也要加

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python visualization/visualize_attention.py \
    --model_path PATH_TO_MODEL \
    --image_dir PATH_TO_IMAGE_DIR \
    --input_file PATH_TO_INFERENCE_RESULT_JSON \
    --output_dir PATH_TO_OUTPUT_DIR \
    --prompt_file prompt.json \
    --model_select Qwen3-VL-8B-Instruct \
    --batch_size 30 \
    --all_word_vis \
    --use_think  # 如果推理时用了 CoT，这里也要加
