cd "$(dirname "$0")/../.."

#JSONL_DIR=/mnt/sdb/wangxinran/zhangxiao/add_Hallu_types/test_data_per_1.jsonl
JSONL_DIR=/mnt/sdc/Hallucination_DATA/Hallucination_Bench/test_gt_withtag_end.jsonl
IMAGE_DIR=/mnt/sdc/Hallucination_DATA/Hallucination-Benchmark/test

CUDA_VISIBLE_DEVICES=1 python run_pipeline.py \
    --mode offline \
    --image_dir $IMAGE_DIR \
    --input_json $JSONL_DIR \
    --model_select seed-2.0-pro \
    --exp_name testset_seed-2.0-pro_think_0319 \
    --use_think \
    --work_dir /mnt/sdb/wangxinran/zhangxiao/test_result 