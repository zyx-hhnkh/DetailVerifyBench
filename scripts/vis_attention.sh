PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python visualization/visualize_attention.py \
    --model_path /mnt/sdc/model_zoo/Qwen3-VL-8B-Instruct \
    --image_dir ./ms_data/test \
    --input_file ./processing_data_zyx/alltest-Qwen3vl8B-nothink/inference_results/tested_model_output_seed_42_filtered.json \
    --output_dir ./processing_data_zyx/alltest-Qwen3vl8B-nothink/visualizations \
    --prompt_file prompt.json \
    --model_select Qwen3-VL-8B-Instruct \
    --max_samples 10 \
    --batch_size 30 \
    --use_think  # 如果推理时用了 CoT，这里也要加

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1 python visualization/visualize_attention.py \
    --model_path /mnt/sdc/model_zoo/Qwen3-VL-8B-Instruct \
    --image_dir ./vis_att_test/images \
    --input_file ./processing_data_zyx/alltest-Qwen3vl8B-nothink/inference_results/tested_model_output_seed_42_filtered.json \
    --output_dir ./vis_att_test/visualizations \
    --prompt_file prompt.json \
    --model_select Qwen3-VL-8B-Instruct \
    --batch_size 30 \
    --all_word_vis \
    --use_think  # 如果推理时用了 CoT，这里也要加