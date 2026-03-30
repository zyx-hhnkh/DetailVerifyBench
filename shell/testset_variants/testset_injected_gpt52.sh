cd "$(dirname "$0")/../.."

JSONL_DIR1=/mnt/sdc/zhangyuxuan/Hallucination_Inject/output/testsets_0318/testset_geminiflash_vs_gpt52_n1.jsonl
JSONL_DIR2=/mnt/sdc/zhangyuxuan/Hallucination_Inject/output/testsets_0318/testset_geminiflash_vs_gpt52_n2.jsonl
JSONL_DIR3=/mnt/sdc/zhangyuxuan/Hallucination_Inject/output/testsets_0318/testset_geminiflash_vs_gpt52_n3.jsonl
IMAGE_DIR=./ms_data/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR1 \
    --model_select gpt-5.2 \
    --exp_name testset0318_injected_n1_gpt52_think_0316 \
    --use_think \
    --work_dir ./ 

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR2 \
    --model_select gpt-5.2 \
    --exp_name testset0318_injected_n2_gpt52_think_0316 \
    --use_think \
    --work_dir ./ 

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode advi \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR3 \
    --model_select gpt-5.2 \
    --exp_name testset0318_injected_n3_gpt52_think_0316 \
    --use_think \
    --work_dir ./ 
