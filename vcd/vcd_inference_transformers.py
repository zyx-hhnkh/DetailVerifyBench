#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCD (Visual Contrastive Decoding) hallucination detection inference.

Uses teacher-forcing to compare per-token log-probabilities under original
vs. Gaussian-noised images. Tokens where the probability difference (delta)
is below a threshold are tagged as hallucinated.

Reference: "Mitigating Object Hallucinations in Large Vision-Language Models
through Visual Contrastive Decoding" (Leng et al., 2023)
"""
import json
import os
import argparse
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference import load_eval_input_data
from vcd.vcd_utils import add_gaussian_noise, aggregate_subtoken_scores, tag_caption_from_scores

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Teacher-forcing prompt (simple description prompt per VCD paper)
TF_USER_PROMPT = "Please describe this image in detail."


def build_teacher_forcing_input(processor, image, caption, device):
    """Build teacher-forcing input for Qwen3-VL models.

    Constructs a conversation where the caption is the assistant's response,
    then tokenizes the full sequence to extract caption token positions.

    Args:
        processor: HuggingFace processor (with tokenizer + image processor).
        image: PIL Image.
        caption: The caption text to score.
        device: torch device.

    Returns:
        inputs: dict of tensors ready for model forward pass.
        caption_start_idx: index of the first caption token in input_ids.
        caption_token_ids: list of token IDs for the caption portion.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": TF_USER_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": caption,
        },
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0]  # [seq_len]

    # Locate caption tokens in the full input sequence.
    # Strategy: tokenize the caption independently, then find where those
    # exact token IDs appear in the full sequence (searching from the end).
    caption_only_ids = processor.tokenizer.encode(caption, add_special_tokens=False)
    n_caption = len(caption_only_ids)

    input_ids_list = input_ids.tolist()
    seq_len = len(input_ids_list)

    # Search for the caption token subsequence from the end of the sequence.
    # This is robust regardless of what the chat template inserts before the
    # caption (e.g. <think>\n\n</think>\n\n for Thinking models).
    caption_start_idx = None
    for i in range(seq_len - n_caption, -1, -1):
        if input_ids_list[i:i + n_caption] == caption_only_ids:
            caption_start_idx = i
            break

    if caption_start_idx is None:
        # Fallback: find the last <|im_end|> and work backwards
        im_end_ids = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if im_end_ids:
            eos_id = im_end_ids[0]
            for i in range(seq_len - 1, -1, -1):
                if input_ids_list[i] == eos_id:
                    caption_start_idx = i - n_caption
                    break

    if caption_start_idx is None:
        caption_start_idx = seq_len - n_caption - 1
        logger.warning("Could not locate caption tokens; using fallback position")

    caption_token_ids = input_ids_list[caption_start_idx:caption_start_idx + n_caption]

    # Verification
    decoded_caption = processor.tokenizer.decode(caption_token_ids, skip_special_tokens=False)
    if decoded_caption.strip() != caption.strip():
        logger.warning(
            f"Caption token mismatch!\n"
            f"  Expected: {caption[:100]}...\n"
            f"  Got:      {decoded_caption[:100]}..."
        )

    return inputs, caption_start_idx, caption_token_ids


def compute_token_logprobs(model, inputs, caption_start, caption_token_ids):
    """Compute per-token log-probabilities for caption tokens via forward pass.

    Uses autoregressive convention: logits at position i predict token i+1.

    Args:
        model: HuggingFace model.
        inputs: dict of input tensors.
        caption_start: start index of caption tokens in the sequence.
        caption_token_ids: list of caption token IDs.

    Returns:
        Tensor of shape [n_caption] with log-probabilities for each caption token.
    """
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # [seq_len, vocab_size]

    n_caption = len(caption_token_ids)

    # Autoregressive shift: logits at position (caption_start - 1 + i) predict
    # the token at position (caption_start + i)
    predict_positions = list(range(caption_start - 1, caption_start - 1 + n_caption))
    predict_logits = logits[predict_positions]  # [n_caption, vocab_size]

    log_probs = F.log_softmax(predict_logits, dim=-1)

    # Gather log-probs for actual caption tokens
    caption_ids_tensor = torch.tensor(caption_token_ids, device=log_probs.device)
    token_logprobs = log_probs[
        torch.arange(n_caption, device=log_probs.device), caption_ids_tensor
    ]

    return token_logprobs


def vcd_detect_single(model, processor, image_path, caption, gamma, device):
    """Run VCD detection on a single item.

    Batches the original and distorted image forward passes together (batch=2)
    for better GPU utilization.

    Args:
        model: HuggingFace model.
        processor: HuggingFace processor.
        image_path: Path to the image file.
        caption: Caption text to score.
        gamma: Gaussian noise intensity.
        device: torch device.

    Returns:
        caption_token_ids: list of token IDs for the caption.
        delta: Tensor of per-token delta scores (logP_orig - logP_distorted).
    """
    img_orig = Image.open(image_path).convert("RGB")
    img_dist = add_gaussian_noise(img_orig, gamma)

    # Build teacher-forcing inputs for both images
    inputs_orig, start_orig, caption_tids = build_teacher_forcing_input(
        processor, img_orig, caption, device
    )
    inputs_dist, start_dist, _ = build_teacher_forcing_input(
        processor, img_dist, caption, device
    )

    n_caption = len(caption_tids)
    caption_ids_tensor = torch.tensor(caption_tids, device=device)

    # Batch the two forward passes (orig + distorted) if sequences have same length
    if inputs_orig["input_ids"].shape == inputs_dist["input_ids"].shape and start_orig == start_dist:
        # Concatenate along batch dimension for a single forward pass
        batched_inputs = {}
        for k in inputs_orig:
            batched_inputs[k] = torch.cat([inputs_orig[k], inputs_dist[k]], dim=0)

        with torch.no_grad():
            outputs = model(**batched_inputs)

        logits = outputs.logits  # [2, seq_len, vocab_size]
        predict_positions = list(range(start_orig - 1, start_orig - 1 + n_caption))

        log_probs = F.log_softmax(logits[:, predict_positions, :], dim=-1)  # [2, n_caption, vocab]
        arange = torch.arange(n_caption, device=device)
        logp_orig = log_probs[0, arange, caption_ids_tensor]
        logp_dist = log_probs[1, arange, caption_ids_tensor]

        del batched_inputs, outputs, logits, log_probs
    else:
        # Fallback: two separate forward passes (different sequence lengths)
        logp_orig = compute_token_logprobs(model, inputs_orig, start_orig, caption_tids)
        logp_dist = compute_token_logprobs(model, inputs_dist, start_dist, caption_tids)
        del inputs_orig, inputs_dist

    # VCD delta: positive = visually grounded, negative/small = language prior (hallucination)
    delta = logp_orig - logp_dist

    torch.cuda.empty_cache()
    return caption_tids, delta


def resolve_image_path(item, image_dir):
    """Resolve the image path from item metadata, same logic as inference.py."""
    raw_image_path = item.get("image_path", "")
    if "./test/" in raw_image_path:
        rel_path = raw_image_path.split("./test/", 1)[1]
    else:
        rel_path = item["id"]
    image_path = os.path.join(image_dir, rel_path)

    if not os.path.exists(image_path):
        filename = os.path.basename(rel_path)
        for sub in os.listdir(image_dir):
            candidate = os.path.join(image_dir, sub, filename)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Image not found: {image_path}")
    return image_path


def save_results_to_disk(final_results, output_filename):
    """Save results to disk (both full and filtered versions)."""
    results_list = list(final_results.values())
    results_list.sort(key=lambda x: x["id"])

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    # Write filtered file
    filtered_results = [r for r in results_list if r.get("validation_passed") is True]
    if output_filename.endswith(".json"):
        filtered_filename = output_filename.replace(".json", "_filtered.json")
    else:
        filtered_filename = output_filename + "_filtered.json"

    with open(filtered_filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)


def auto_resume(all_items, output_filename):
    """Resume from existing results, same pattern as inference.py."""
    final_results = {}
    if os.path.exists(output_filename):
        logger.info(f"Found existing output file: {output_filename}")
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            if isinstance(existing_data, list):
                for res in existing_data:
                    if "id" in res:
                        final_results[res["id"]] = res
                logger.info(f"Loaded {len(final_results)} completed items from checkpoint")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}. Will rerun all data.")
            final_results = {}

    processed_ids = set(final_results.keys())
    pending_items = [item for item in all_items if item["id"] not in processed_ids]

    if len(pending_items) == 0:
        logger.info("All data has been processed, skipping inference")
    else:
        logger.info(f"Remaining: {len(pending_items)} / {len(all_items)}")

    return final_results, pending_items


def run_vcd(model, processor, all_items, image_dir, output_filename, gamma, threshold, device):
    """Main VCD inference loop.

    Args:
        model: HuggingFace model.
        processor: HuggingFace processor.
        all_items: List of data items from load_eval_input_data.
        image_dir: Image directory path.
        output_filename: Output JSON file path.
        gamma: Gaussian noise intensity.
        threshold: Delta threshold for hallucination tagging.
        device: torch device.
    """
    final_results, pending_items = auto_resume(all_items, output_filename)

    if not pending_items:
        return

    tokenizer = processor.tokenizer

    for item in tqdm(pending_items, desc="VCD Detection"):
        item_id = item["id"]

        try:
            image_path = resolve_image_path(item, image_dir)
        except FileNotFoundError as e:
            logger.error(f"[{item_id}] {e}")
            continue

        # Determine which caption to score
        is_use_original = item.get("use_original_caption_flag", False)
        caption = item["original_caption"] if is_use_original else item["hallucinated_caption"]

        try:
            caption_tids, delta = vcd_detect_single(
                model, processor, image_path, caption, gamma, device
            )

            # Aggregate subtoken scores to word level
            words_scores = aggregate_subtoken_scores(
                caption_tids, delta, tokenizer, method="mean"
            )

            # Generate tagged output
            tagged_output = tag_caption_from_scores(words_scores, threshold, caption)

            # Build debug info: per-word delta scores
            debug_scores = [
                {"word": w, "delta": round(d, 4)} for w, d in words_scores
            ]

            result = {
                "id": item_id,
                "image_path": item["image_path"],
                "original_caption": item["original_caption"],
                "hallucinated_caption": item["hallucinated_caption"],
                "gt_hallucinated_caption_with_tags": item.get("hallucinated_caption_with_tags", ""),
                "hallucinated_caption_with_tags": tagged_output,
                "data_source": item.get("data_source", "empty"),
                "is_use_original": is_use_original,
                "validation_passed": True,  # VCD always produces valid format
                "validation_message": None,
                "attempt": 1,
                "seed": 0,
                "raw_output": json.dumps(debug_scores, ensure_ascii=False),
                "vcd_gamma": gamma,
                "vcd_threshold": threshold,
            }

        except Exception as e:
            logger.error(f"[{item_id}] VCD detection failed: {e}")
            import traceback
            traceback.print_exc()
            result = {
                "id": item_id,
                "image_path": item["image_path"],
                "original_caption": item["original_caption"],
                "hallucinated_caption": item["hallucinated_caption"],
                "gt_hallucinated_caption_with_tags": item.get("hallucinated_caption_with_tags", ""),
                "hallucinated_caption_with_tags": "",
                "data_source": item.get("data_source", "empty"),
                "is_use_original": item.get("use_original_caption_flag", False),
                "validation_passed": False,
                "validation_message": f"VCD error: {str(e)}",
                "attempt": 1,
                "seed": 0,
                "raw_output": "",
            }

        final_results[item_id] = result
        save_results_to_disk(final_results, output_filename)

    # Final summary
    results_list = list(final_results.values())
    pass_count = sum(1 for r in results_list if r["validation_passed"])
    logger.info(f"VCD completed! Total: {len(results_list)}, Passed: {pass_count}")


def main():
    parser = argparse.ArgumentParser(
        description="VCD (Visual Contrastive Decoding) Hallucination Detection"
    )
    parser.add_argument("--model_path", required=True, help="Path to Transformers model")
    parser.add_argument("--model_select", required=True, help="Model name (e.g., Qwen3-VL-8B-Thinking)")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--input_dir", required=True, help="Formatted JSON input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--vcd_gamma", type=float, default=0.1,
                        help="Gaussian noise intensity (default: 0.1)")
    parser.add_argument("--vcd_threshold", type=float, default=0.0,
                        help="Delta threshold for hallucination tagging (default: 0.0)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit items per category (for testing)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    all_items = load_eval_input_data(args.input_dir, limit_per_category=args.limit)

    # Load model
    logger.info(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    device = next(model.parameters()).device
    logger.info(f"Model loaded on device: {device}")

    # Output file (seed=0, VCD is deterministic)
    output_filename = os.path.join(args.output_dir, "tested_model_output_seed_0.json")

    # Run VCD inference
    run_vcd(
        model=model,
        processor=processor,
        all_items=all_items,
        image_dir=args.image_dir,
        output_filename=output_filename,
        gamma=args.vcd_gamma,
        threshold=args.vcd_threshold,
        device=device,
    )

    logger.info("VCD inference completed!")


if __name__ == "__main__":
    main()
