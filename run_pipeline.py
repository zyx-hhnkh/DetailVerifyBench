#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main pipeline controller for DOCCI hallucination detection.
Updated to align with multi-seed inference and integrated metric calculation.
"""
import os
import sys
import argparse
import subprocess
from datetime import datetime


# SEEDS = [42, 94, 2025] #不动了，之后可以不重新跑
SEEDS = [42]

def run_step(script, args_list):
    """Helper function to run a subprocess step with logging."""
    print(f"\n{'='*80}")
    print(f">>> Running {os.path.basename(script)}")
    print("Command:", " ".join([sys.executable, script] + args_list))
    
    try:
        subprocess.run([sys.executable, script] + args_list, check=True)
        print(f"✅ {os.path.basename(script)} finished successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {os.path.basename(script)}: {e}")
        sys.exit(1)

def main():
    global SEEDS
    parser = argparse.ArgumentParser(description="Run the full hallucination detection pipeline.")
    
    # 基础路径参数
    parser.add_argument("--image_dir", required=True, help="Directory containing images.")
    parser.add_argument("--model_path", help="Path to the local LLM model.")
    parser.add_argument("--input_json", required=True, nargs='+', help="Input JSONL file (online) or Directory (offline).")
    parser.add_argument("--work_dir", required=True, help="Base directory for outputs.")
    
    # 实验配置
    parser.add_argument("--exp_name", default="exp", help="Name of this experiment. Used for folder naming.")
    parser.add_argument("--use_think", action="store_true", help="Enable Chain-of-Thought (CoT) prompting in inference.")
    
    # 流程控制
    # ⭐修改：减少了步骤，移除了独立的 filter 和 to01 步骤，因为它们已被集成
    parser.add_argument("--skip", nargs="*", default=[], choices=["format", "infer", "metrics"],
                        help="Steps to skip.")
    parser.add_argument("--mode", default="offline", choices=["offline", "CD", "advi", "advi_nodetect", "tldr"],
                        help="Data processing mode: offline (skip format), CD (Constructive-Decoding), advi (adversarial injection), advi_nodetect (adversarial, use description_tag_nodetect), tldr (TLDR testset inject)")
    parser.add_argument("--model_select", required=True, help="model select")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N items per category (for quick testing)")
    parser.add_argument("--api_concurrency", type=int, default=10, help="Number of parallel API calls (default: 10)")

    # VCD Plugin
    parser.add_argument("--plugin", default=None, choices=["VCD"],
                        help="Use a plugin for inference. VCD = Visual Contrastive Decoding.")
    parser.add_argument("--vcd_gamma", type=float, default=0.1,
                        help="VCD: Gaussian noise intensity (default: 0.1)")
    parser.add_argument("--vcd_threshold", type=float, default=0.0,
                        help="VCD: Delta threshold for hallucination tagging (default: 0.0)")
    parser.add_argument("--shard", type=str, default=None,
                        help="VCD: Data sharding for multi-GPU: 'i/n' (e.g., '0/2')")
    args = parser.parse_args()

    # ⭐ 1. 生成自动时间戳实验目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 不加时间戳了！！！同一个实验能断点续跑
    exp_folder_name = f"{args.exp_name}"
    
    # 获取脚本所在目录
    root = os.path.dirname(os.path.abspath(__file__))
    
    # 构建目录结构
    ############################## 关键目录 ####################
    base_proc_dir = os.path.join(args.work_dir, "processing_data_zyx_260317")
    proc_dir = os.path.join(base_proc_dir, exp_folder_name)
    os.makedirs(proc_dir, exist_ok=True)
    
    # 1. 格式化后的数据目录 (Gold Data Directory)
    formatted_data_dir = os.path.join(proc_dir, "formatted_jsons")
    # 2. 推理结果目录
    inference_result_dir = os.path.join(proc_dir, "inference_results")
    # 3. 最终指标输出路径
    final_metrics_file = os.path.join(proc_dir, "final_averaged_metrics.json")
    # 4. Gold 数据处理的中间文件 (Step 5 需要)
    gold_processed_tmp = os.path.join(proc_dir, "golden_file_alter_to_01.json")

    # 确保目录存在
    os.makedirs(formatted_data_dir, exist_ok=True)
    os.makedirs(inference_result_dir, exist_ok=True)

    print(f"\n📁 Outputs directory: {args.work_dir}")
    print(f"📂 Experiment save directory: {proc_dir}")
    print(f"⏰ Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔧 Configuration: Use Think={args.use_think}, Mode={args.mode}")

# ================= Step 1: Data Formatting =================
    if args.mode == "offline":
        print("\n🟧 [Mode: Offline] Running JSONL data formatting...")
        if "format" not in args.skip:
            format_args = ["--input_jsonls"] + args.input_json + ["--output_dir", formatted_data_dir, "--mode", "offline"]
            run_step(os.path.join(root, "alter_data_format.py"), format_args)
        infer_input_dir = formatted_data_dir
    elif args.mode == "CD":
        print("\n🟦 [Mode: CD] Running Constructive-Decoding data formatting...")
        if "format" not in args.skip:
            format_args = ["--input_jsonls"] + args.input_json + ["--output_dir", formatted_data_dir, "--mode", "CD"]
            run_step(os.path.join(root, "alter_data_format.py"), format_args)
        infer_input_dir = formatted_data_dir
    elif args.mode == "advi":
        print("\n🟪 [Mode: advi] Running adversarial injection data formatting...")
        if "format" not in args.skip:
            format_args = ["--input_jsonls"] + args.input_json + ["--output_dir", formatted_data_dir, "--mode", "advi"]
            run_step(os.path.join(root, "alter_data_format.py"), format_args)
        infer_input_dir = formatted_data_dir
    elif args.mode == "advi_nodetect":
        print("\n🟪 [Mode: advi_nodetect] Running adversarial injection (nodetect) data formatting...")
        if "format" not in args.skip:
            format_args = ["--input_jsonls"] + args.input_json + ["--output_dir", formatted_data_dir, "--mode", "advi_nodetect"]
            run_step(os.path.join(root, "alter_data_format.py"), format_args)
        infer_input_dir = formatted_data_dir
    elif args.mode == "tldr":
        print("\n🟩 [Mode: tldr] Running TLDR testset inject data formatting...")
        if "format" not in args.skip:
            format_args = ["--input_jsonls"] + args.input_json + ["--output_dir", formatted_data_dir, "--mode", "tldr"]
            run_step(os.path.join(root, "alter_data_format.py"), format_args)
        infer_input_dir = formatted_data_dir

    # ================= Step 2: Inference (Multi-Seed) =================
    # VCD plugin overrides SEEDS to [0] (deterministic, no multi-seed)
    if args.plugin == "VCD":
        SEEDS = [0]

    if "infer" not in args.skip:
        if args.plugin == "VCD":
            # VCD plugin: use vcd_inference.py instead of inference.py
            vcd_args = [
                "--image_dir", args.image_dir,
                "--input_dir", infer_input_dir,
                "--output_dir", inference_result_dir,
                "--model_select", args.model_select,
                "--vcd_gamma", str(args.vcd_gamma),
                "--vcd_threshold", str(args.vcd_threshold),
            ]
            if args.model_path:
                vcd_args.extend(["--model_path", args.model_path])
            if args.limit is not None:
                vcd_args.extend(["--limit", str(args.limit)])
            if args.shard is not None:
                vcd_args.extend(["--shard", args.shard])
            run_step(os.path.join(root, "vcd", "vcd_inference_transformers.py"), vcd_args)
        else:
            # Standard inference
            infer_args = [
                "--image_dir", args.image_dir,
                "--input_dir", infer_input_dir,
                "--output_dir", inference_result_dir,
                "--model_select", args.model_select,
                "--seeds"
            ]

            infer_args.extend([str(s) for s in SEEDS])

            if args.model_path:
                infer_args.extend(["--model_path", args.model_path])
            if args.use_think:
                infer_args.append("--use_think")
            if args.limit is not None:
                infer_args.extend(["--limit", str(args.limit)])
            if args.api_concurrency != 10:
                infer_args.extend(["--api_concurrency", str(args.api_concurrency)])

            run_step(os.path.join(root, "inference.py"), infer_args)

    # ================= Step 3: Metrics Calculation =================
    if "metrics" not in args.skip:
        
        # 自动寻找推理步骤生成的 3 个 filtered 文件
        test_files_list = []
        for seed in SEEDS:
            # 文件名格式参考 inference 脚本: tested_model_output_seed_{seed}_filtered.json
            filename = f"tested_model_output_seed_{seed}_filtered.json"
            filepath = os.path.join(inference_result_dir, filename)
            
            if os.path.exists(filepath):
                test_files_list.append(filepath)
            else:
                print(f"⚠️ Warning: Expected result file not found: {filepath}")

        if not test_files_list:
            print("❌ Error: No filtered result files found for metrics calculation.")
            sys.exit(1)

        metrics_args = [
            "--input_gold_dir", infer_input_dir,      # Gold 数据目录 (Step 1 的输出)
            "--output_gold_processed", gold_processed_tmp, # 中间文件路径
            "--output_metrics", final_metrics_file,   # 最终结果
            "--image_dir", args.image_dir,            # 图片目录，用于按领域统计
            "--input_test_files"                      # 后面接多个文件路径
        ] + test_files_list

        run_step(os.path.join(root, "calculate_metrics.py"), metrics_args)

    print(f"\n🎉 All steps completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📄 Final Metrics saved to: {final_metrics_file}")
    print(f"📂 Full Experiment Directory: {proc_dir}")

if __name__ == "__main__":
    main()