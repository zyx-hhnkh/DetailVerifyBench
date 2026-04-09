# DetailVerifyBench

A comprehensive evaluation framework for detecting hallucinations in Vision-Language Models (VLMs). This benchmark measures how accurately models can identify false or incorrect descriptions in image captions at the token level.

## Features

- **Multi-model support**: Qwen3-VL, GPT-5.x, Gemini-3.x, Claude Opus, Step-3-VL, Kimi, Seed, and more
- **Multiple inference backends**: vLLM (batch), Transformers, OpenAI API, Google Gemini API, Anthropic API
- **VCD (Visual Contrastive Decoding)**: Alternative hallucination detection method based on contrastive log-probability analysis
- **Token-level metrics**: Precision, Recall, F1, and IoU at both token and span levels
- **Multi-seed evaluation**: Reproducible results with configurable random seeds
- **Visualization tools**: Attention heatmaps and Gradio-based result viewer

## Installation

```bash
pip install -r requirements.txt
```

For local model inference, install [vLLM](https://docs.vllm.ai/en/latest/getting_started/quickstart/).

Download NLTK data for sentence tokenization:
```python
import nltk
nltk.download('punkt_tab')
```

## Data Preparation

Download the dataset from HuggingFace:

```bash
huggingface-cli download zyxhhnkh/DetailVerifyBench \
    --repo-type dataset \
    --local-dir     ./data
```

## Usage

### Real Hallucination (Local Model)

```bash
python run_pipeline.py \
    --mode          offline \
    --image_dir     PATH_TO_test_set \
    --input_json    PATH_TO_real_jsonl \
    --model_select  "model_name" \
    --model_path    "PATH_TO_model_name" \
    --exp_name      "my_experiment" \
    --work_dir      ./ \
    --use_think
```

### Real Hallucination (API-based)

```bash
python run_pipeline.py \
    --mode          offline \
    --image_dir     PATH_TO_test_set \
    --input_json    PATH_TO_real_jsonl \
    --model_path    "PATH_TO_model_name" \
    --exp_name      "my_experiment" \
    --work_dir      ./ \
    --use_think
```

### Synthetic Hallucination (Local Model)

```bash
python run_pipeline.py \
    --mode          advi \
    --image_dir     PATH_TO_test_set \
    --input_json    PATH_TO_synthetic_jsonl \
    --model_select  "model_name" \
    --model_path    "PATH_TO_model_name" \
    --exp_name      "my_experiment" \
    --work_dir      ./ \
    --use_think
```

### Synthetic Hallucination (API-based)

```bash
python run_pipeline.py \
    --mode          advi \
    --image_dir     PATH_TO_test_set \
    --input_json    PATH_TO_synthetic_jsonl \
    --model_select  "model_name" \
    --exp_name      "my_experiment" \
    --work_dir      ./ \
    --use_think
```

### Using Shell Scripts

Pre-configured evaluation scripts are available in `shell/`:

```bash
# Standard testset evaluation
bash shell/eval_testset/eval_testset_qwen3vl8bthink.sh

# Adversarial injection evaluation
bash shell/eval_injected_testset/eval_injected_testset_qwen3vl8bthink.sh
```

### VCD Mode

```bash
bash shell/eval_testset/eval_testset_qwen3vl8bthink_vcd.sh
```

### Visualization

```bash
# Attention heatmap visualization
bash scripts/vis_attention.sh

# Gradio-based result viewer
bash scripts/vis_result.sh
```

## Project Structure

```
├── run_pipeline.py              # Main pipeline orchestrator
├── inference.py                 # Multi-backend inference engine
├── calculate_metrics.py         # Token/span-level metric computation
├── alter_data_format.py         # JSONL ↔ JSON format converter
├── prompt.py                    # Model-specific prompt templates
├── requirements.txt             # Python dependencies
│
├── vcd/                         # Visual Contrastive Decoding
│   ├── vcd_inference_transformers.py
│   ├── vcd_inference_vllm.py
│   └── vcd_utils.py
│
├── visualization/               # Visualization tools
│   ├── visualize_attention.py   # Attention heatmap generation
│   ├── visualize_results.py     # Gradio result viewer
│   ├── Data_To_Chart.py         # LaTeX table generation
│   └── gen_inject_table.py      # Injection comparison tables
│
├── scripts/                     # Orchestration scripts
│   ├── start.sh                 # High-level experiment launcher
│   ├── run_experiments.sh       # Multi-experiment runner
│   ├── vis_attention.sh
│   ├── vis_attention_perlayer.sh
│   └── vis_result.sh
│
├── shell/                       # Evaluation launch scripts
│   ├── eval_testset/            # Standard testset evaluations
│   ├── eval_injected_testset/   # Adversarial injection evaluations
│   └── testset_variants/        # Other test configurations
│
├── tests/                       # Testing utilities
│   └── test_api.py              # API connectivity tests
│
└── data/                        # Dataset (download separately)
```

## Pipeline Overview

```
Input: test_gt_withtag.jsonl + images
         │
         ▼
  alter_data_format.py    (JSONL → individual JSON files)
         │
         ▼
  inference.py            (Model inference with hallucination tagging)
         │
         ▼
  calculate_metrics.py    (Precision, Recall, F1, IoU computation)
         │
         ▼
  Output: final_averaged_metrics.json
```

## Supported Models

| Model | Backend | Config Flag |
|-------|---------|-------------|
| Qwen3-VL-8B | vLLM / Transformers | `Qwen3-VL-8B` |
| Qwen3-VL-4B | vLLM / Transformers | `Qwen3-VL-4B` |
| GPT-5.2 / 5.4 | OpenAI API | `gpt-5.2` / `gpt-5.4` |
| Gemini-3-Pro / 3.1-Pro | Google API | `gemini-3-pro` / `gemini-3.1-pro` |
| Claude Opus 4.6 | Anthropic API | `opus-4.6` |
| Step-3-VL-10B | vLLM | `Step-3-VL-10B` |
| Kimi-K2.5 | OpenRouter API | `kimi-k2.5` |
| Seed-2.0-Pro | API | `seed-2.0-pro` |

## Environment Variables

For API-based models, configure the following in a `.env` file:

```bash
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
OPENROUTER_API_KEY=your_key
```

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{wang2026detailverifybench,
      title={DetailVerifyBench: A Benchmark for Dense Hallucination Localization in Long Image Captions}, 
      author={Xinran Wang and Yuxuan Zhang and Xiao Zhang and Haolong Yan and Muxi Diao and Songyu Xu and Zhonghao Yan and Hongbing Li and Kongming Liang and Zhanyu Ma},
      year={2026},
      eprint={2604.05623},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

[To be determined]
