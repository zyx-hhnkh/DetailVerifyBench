#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCD (Visual Contrastive Decoding) hallucination detection inference.
vLLM version: uses prompt_logprobs for high-throughput teacher-forcing.

Uses vLLM's batch scheduling + PagedAttention for efficient GPU utilization.
Compares per-token log-probabilities under original vs. Gaussian-noised images.
Tokens where the probability difference (delta) is below a threshold are tagged
as hallucinated.

Reference: "Mitigating Object Hallucinations in Large Vision-Language Models
through Visual Contrastive Decoding" (Leng et al., 2023)
"""
import json
import io
import os
import base64
import argparse
import logging
from tqdm import tqdm
from PIL import Image
from vllm import LLM, SamplingParams

from inference import load_eval_input_data
from vcd.vcd_utils import add_gaussian_noise, aggregate_subtoken_scores, tag_caption_from_scores

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Teacher-forcing prompt
TF_USER_PROMPT = "Please describe this image in detail."

# Batch size for vLLM calls (number of items, each produces 2 requests)
VCD_BATCH_SIZE = 4


def encode_pil_to_base64(pil_image):
    """Encode a PIL Image to base64 string (JPEG format)."""
    buf = io.BytesIO()
    img = pil_image
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def build_vcd_messages(image_b64, caption):
    """Build teacher-forcing messages with caption as assistant response."""
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": TF_USER_PROMPT},
            ],
        },
        {
            "role": "assistant",
            "content": caption,
        },
    ]


def extract_caption_logprobs(output, tokenizer, caption):
    """Extract per-token logprobs for the caption portion from vLLM output.

    Args:
        output: vLLM RequestOutput with prompt_logprobs.
        tokenizer: The tokenizer (from llm.get_tokenizer()).
        caption: The caption text.

    Returns:
        caption_token_ids: list of token IDs for the caption.
        logprobs: list of float logprobs for each caption token.
    """
    prompt_token_ids = list(output.prompt_token_ids)
    caption_only_ids = tokenizer.encode(caption, add_special_tokens=False)
    n_caption = len(caption_only_ids)

    # Search for caption token subsequence from the end
    caption_start = None
    for i in range(len(prompt_token_ids) - n_caption, -1, -1):
        if prompt_token_ids[i:i + n_caption] == caption_only_ids:
            caption_start = i
            break

    if caption_start is None:
        raise ValueError(
            f"Cannot locate caption tokens in prompt. "
            f"Caption has {n_caption} tokens, prompt has {len(prompt_token_ids)} tokens."
        )

    # Extract logprob for each caption token
    # prompt_logprobs[i] contains the logprob of token at position i
    # (predicted by context up to position i-1)
    logprobs = []
    for j in range(caption_start, caption_start + n_caption):
        pos_logprobs = output.prompt_logprobs[j]
        if pos_logprobs is None:
            logprobs.append(0.0)
            continue
        actual_token_id = caption_only_ids[j - caption_start]
        if actual_token_id in pos_logprobs:
            logprobs.append(pos_logprobs[actual_token_id].logprob)
        else:
            # Token not in top-K logprobs; use a very negative value
            logprobs.append(-100.0)

    return caption_only_ids, logprobs


def resolve_image_path(item, image_dir):
    """Resolve the image path from item metadata."""
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

    filtered_results = [r for r in results_list if r.get("validation_passed") is True]
    if output_filename.endswith(".json"):
        filtered_filename = output_filename.replace(".json", "_filtered.json")
    else:
        filtered_filename = output_filename + "_filtered.json"

    with open(filtered_filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)


def auto_resume(all_items, output_filename):
    """Resume from existing results."""
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

    if not pending_items:
        logger.info("All data has been processed, skipping inference")
    else:
        logger.info(f"Remaining: {len(pending_items)} / {len(all_items)}")

    return final_results, pending_items


def run_vcd(llm, tokenizer, all_items, image_dir, output_filename, gamma, threshold):
    """Main VCD inference loop using vLLM batch processing.

    Args:
        llm: vLLM LLM instance.
        tokenizer: Tokenizer from llm.get_tokenizer().
        all_items: List of data items.
        image_dir: Image directory path.
        output_filename: Output JSON file path.
        gamma: Gaussian noise intensity.
        threshold: Delta threshold for hallucination tagging.
    """
    final_results, pending_items = auto_resume(all_items, output_filename)
    if not pending_items:
        return

    sampling_params = SamplingParams(
        prompt_logprobs=1,  # Return logprob of each prompt token
        max_tokens=1,       # Don't generate; only extract prompt logprobs
        temperature=0.0,
    )

    # Process in batches
    for batch_start in range(0, len(pending_items), VCD_BATCH_SIZE):
        batch_items = pending_items[batch_start:batch_start + VCD_BATCH_SIZE]
        batch_end = min(batch_start + VCD_BATCH_SIZE, len(pending_items))
        logger.info(f"Processing batch {batch_start}-{batch_end} / {len(pending_items)}")

        # Prepare all messages for this batch (orig + dist for each item)
        all_messages = []
        batch_captions = []
        batch_valid = []  # Track which items were successfully prepared

        for item in batch_items:
            item_id = item["id"]
            try:
                image_path = resolve_image_path(item, image_dir)
                img_orig = Image.open(image_path).convert("RGB")
                img_dist = add_gaussian_noise(img_orig, gamma)

                is_use_original = item.get("use_original_caption_flag", False)
                caption = item["original_caption"] if is_use_original else item["hallucinated_caption"]

                b64_orig = encode_pil_to_base64(img_orig)
                b64_dist = encode_pil_to_base64(img_dist)

                all_messages.append(build_vcd_messages(b64_orig, caption))
                all_messages.append(build_vcd_messages(b64_dist, caption))
                batch_captions.append(caption)
                batch_valid.append(item)
            except Exception as e:
                logger.error(f"[{item_id}] Failed to prepare input: {e}")
                final_results[item_id] = {
                    "id": item_id,
                    "image_path": item["image_path"],
                    "original_caption": item["original_caption"],
                    "hallucinated_caption": item["hallucinated_caption"],
                    "gt_hallucinated_caption_with_tags": item.get("hallucinated_caption_with_tags", ""),
                    "hallucinated_caption_with_tags": "",
                    "data_source": item.get("data_source", "empty"),
                    "is_use_original": item.get("use_original_caption_flag", False),
                    "validation_passed": False,
                    "validation_message": f"Preparation error: {str(e)}",
                    "attempt": 1, "seed": 0, "raw_output": "",
                }

        if not all_messages:
            save_results_to_disk(final_results, output_filename)
            continue

        # Run vLLM batch inference with OOM retry (halve batch on failure)
        logger.info(f"Running vLLM inference on {len(all_messages)} requests "
                     f"({len(batch_valid)} items x 2)...")
        outputs = llm.chat(
            messages=all_messages,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        # Process outputs: every 2 consecutive outputs correspond to one item
        for i, item in enumerate(batch_valid):
            item_id = item["id"]
            caption = batch_captions[i]
            is_use_original = item.get("use_original_caption_flag", False)

            try:
                output_orig = outputs[2 * i]
                output_dist = outputs[2 * i + 1]

                caption_tids, logprobs_orig = extract_caption_logprobs(
                    output_orig, tokenizer, caption
                )
                _, logprobs_dist = extract_caption_logprobs(
                    output_dist, tokenizer, caption
                )

                # VCD delta: logP(orig) - logP(distorted)
                delta = [lo - ld for lo, ld in zip(logprobs_orig, logprobs_dist)]

                # Aggregate subtoken scores to word level
                words_scores = aggregate_subtoken_scores(
                    caption_tids, delta, tokenizer, method="mean"
                )

                # Generate tagged output
                tagged_output = tag_caption_from_scores(words_scores, threshold, caption)

                # Debug info
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
                    "validation_passed": True,
                    "validation_message": None,
                    "attempt": 1,
                    "seed": 0,
                    "raw_output": json.dumps(debug_scores, ensure_ascii=False),
                    "vcd_gamma": gamma,
                    "vcd_threshold": threshold,
                }
            except Exception as e:
                logger.error(f"[{item_id}] VCD scoring failed: {e}")
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
                    "attempt": 1, "seed": 0, "raw_output": "",
                }

            final_results[item_id] = result

        # Save after each batch
        save_results_to_disk(final_results, output_filename)
        logger.info(f"Batch saved. Total processed: {len(final_results)}")

    # Final summary
    results_list = list(final_results.values())
    pass_count = sum(1 for r in results_list if r["validation_passed"])
    logger.info(f"VCD completed! Total: {len(results_list)}, Passed: {pass_count}")


def main():
    parser = argparse.ArgumentParser(
        description="VCD (Visual Contrastive Decoding) Hallucination Detection (vLLM)"
    )
    parser.add_argument("--model_path", required=True, help="Path to model")
    parser.add_argument("--model_select", required=True, help="Model name")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--input_dir", required=True, help="Formatted JSON input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--vcd_gamma", type=float, default=0.1,
                        help="Gaussian noise intensity (default: 0.1)")
    parser.add_argument("--vcd_threshold", type=float, default=0.0,
                        help="Delta threshold for hallucination tagging (default: 0.0)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (number of items per vLLM call, default: 4)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit items per category (for testing)")
    args = parser.parse_args()

    global VCD_BATCH_SIZE
    VCD_BATCH_SIZE = args.batch_size

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    all_items = load_eval_input_data(args.input_dir, limit_per_category=args.limit)

    # Load model with vLLM
    logger.info(f"Loading model with vLLM: {args.model_path}")
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 1},
        gpu_memory_utilization=0.95,
    )
    tokenizer = llm.get_tokenizer()
    logger.info("Model loaded successfully")

    # Output file
    output_filename = os.path.join(args.output_dir, "tested_model_output_seed_0.json")

    # Run VCD inference
    run_vcd(
        llm=llm,
        tokenizer=tokenizer,
        all_items=all_items,
        image_dir=args.image_dir,
        output_filename=output_filename,
        gamma=args.vcd_gamma,
        threshold=args.vcd_threshold,
    )

    logger.info("VCD inference completed!")


if __name__ == "__main__":
    main()
