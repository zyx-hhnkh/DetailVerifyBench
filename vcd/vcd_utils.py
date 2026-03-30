#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VCD (Visual Contrastive Decoding) utility functions.
- Image distortion with Gaussian noise
- Subword-to-word score aggregation
- Threshold-based hallucination tagging
"""
import math
import re
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image


def add_gaussian_noise(image: Image.Image, gamma: float = 0.1) -> Image.Image:
    """Apply Gaussian noise to an image following VCD's diffusion forward process.

    noised = sqrt(1 - gamma) * image + sqrt(gamma) * N(0, 1)

    Args:
        image: PIL Image (RGB).
        gamma: Noise intensity in [0, 1]. Default 0.1 per VCD paper.

    Returns:
        Noised PIL Image (RGB).
    """
    img_tensor = to_tensor(image)  # [C, H, W], float32 in [0, 1]
    noise = torch.randn_like(img_tensor)
    noised = math.sqrt(1 - gamma) * img_tensor + math.sqrt(gamma) * noise
    noised = noised.clamp(0, 1)
    return to_pil_image(noised)


def aggregate_subtoken_scores(token_ids, delta_scores, tokenizer, method="mean"):
    """Aggregate subword token scores into word-level scores.

    Args:
        token_ids: List[int] - caption token IDs.
        delta_scores: Tensor or List[float] - per-token delta scores.
        tokenizer: HuggingFace tokenizer.
        method: Aggregation method ("mean", "min", "max").

    Returns:
        List of (word_string, aggregated_delta) tuples.
    """
    if isinstance(delta_scores, torch.Tensor):
        delta_scores = delta_scores.cpu().tolist()

    token_strs = tokenizer.convert_ids_to_tokens(token_ids)

    # Special tokens to skip
    special_prefixes = ("<|", "<image>", "<img>", "<vision>", "</")

    words = []       # List of (word_str, [delta_values])
    current_word = ""
    current_deltas = []

    for i, (tid, tok_str, delta) in enumerate(zip(token_ids, token_strs, delta_scores)):
        # Skip special tokens
        if tok_str is None or any(tok_str.startswith(sp) for sp in special_prefixes):
            continue
        if tid in tokenizer.all_special_ids:
            continue

        # Decode the single token to get its actual text
        decoded = tokenizer.decode([tid])

        # Detect word boundary: leading space or SentencePiece underscore
        is_new_word = (
            decoded.startswith(" ") or
            tok_str.startswith("▁") or  # U+2581, SentencePiece marker
            tok_str.startswith("Ġ") or  # GPT-2 style marker
            i == 0  # first token always starts a new word
        )

        if is_new_word and current_word:
            # Save the previous word
            words.append((current_word, current_deltas))
            current_word = decoded.lstrip(" ").lstrip("▁")
            current_deltas = [delta]
        elif is_new_word:
            current_word = decoded.lstrip(" ").lstrip("▁")
            current_deltas = [delta]
        else:
            current_word += decoded
            current_deltas.append(delta)

    # Don't forget the last word
    if current_word:
        words.append((current_word, current_deltas))

    # Aggregate subtoken scores
    result = []
    for word_str, deltas in words:
        if not deltas:
            continue
        if method == "mean":
            agg = sum(deltas) / len(deltas)
        elif method == "min":
            agg = min(deltas)
        elif method == "max":
            agg = max(deltas)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        result.append((word_str, agg))

    return result


def tag_caption_from_scores(words_scores, threshold, original_caption):
    """Generate tagged caption based on VCD delta scores.

    Words with delta < threshold are marked as hallucinated.
    Consecutive hallucinated words are merged into a single span.

    Args:
        words_scores: List of (word_str, delta) from aggregate_subtoken_scores.
        threshold: Delta threshold. Words below this are hallucinated.
        original_caption: The original caption text (used for reconstruction).

    Returns:
        Tagged caption string, or "NO HALLUCINATION" if all words are above threshold.
    """
    if not words_scores:
        return "NO HALLUCINATION"

    # Mark each word as hallucinated or not
    is_hallucinated = [delta < threshold for _, delta in words_scores]

    # If nothing is hallucinated, return NO HALLUCINATION
    if not any(is_hallucinated):
        return "NO HALLUCINATION"

    # Reconstruct the caption by matching VCD words to original caption positions
    # Use a greedy left-to-right scan through the original caption
    result_parts = []
    pos = 0  # current position in original_caption
    in_hallucination = False

    for i, (word_str, _) in enumerate(words_scores):
        h = is_hallucinated[i]

        # Find this word in the original caption starting from pos
        # Match by finding the word text (case-sensitive)
        word_idx = original_caption.find(word_str, pos)

        if word_idx == -1:
            # Fallback: try case-insensitive or partial match
            # This handles minor tokenizer normalization differences
            lower_idx = original_caption.lower().find(word_str.lower(), pos)
            if lower_idx != -1:
                word_idx = lower_idx
            else:
                # Last resort: just advance past whitespace and take next chunk
                word_idx = pos

        # Add any whitespace/punctuation between previous word and this word
        if word_idx > pos:
            gap = original_caption[pos:word_idx]
            if in_hallucination and h:
                # Both in hallucination span, include the gap inside the span
                result_parts.append(gap)
            elif in_hallucination and not h:
                # End hallucination span, then add gap
                result_parts.append("</HALLUCINATION>")
                result_parts.append(gap)
                in_hallucination = False
            else:
                result_parts.append(gap)

        # Start/end hallucination tags as needed
        if h and not in_hallucination:
            result_parts.append("<HALLUCINATION>")
            in_hallucination = True
        elif not h and in_hallucination:
            result_parts.append("</HALLUCINATION>")
            in_hallucination = False

        # Add the word text from the original caption
        word_end = word_idx + len(word_str)
        actual_word = original_caption[word_idx:word_end]
        result_parts.append(actual_word)
        pos = word_end

    # Close any open hallucination tag
    if in_hallucination:
        result_parts.append("</HALLUCINATION>")

    # Add any trailing text from original caption
    if pos < len(original_caption):
        result_parts.append(original_caption[pos:])

    return "".join(result_parts)
