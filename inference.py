#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run model hallucination tagging inference using vLLM.
Features: 
- 3 Fixed Random Seeds
- Batch Inference
- Automatic Retry Logic
"""
import json
import re
import os
import argparse
import base64
import time
import logging
import concurrent.futures
import torch
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
from google import genai
from google.genai import types
from typing import Tuple
from vllm import LLM
from openai import OpenAI
import anthropic
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from prompt import (
    PROMPT_TEMPLATES,
    get_sampling_params,
    get_transformers_generate_kwargs,
    create_prompt
)

# if there are new models, add them here
VLLM_DEPLOY_MODELS = [
    "Qwen2.5-VL-7B",
    "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Thinking",
    "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-8B-Thinking",
    "ours",
    "Llama-3.2-11B-Vision-Instruct",
    "Step3-VL-10B",
]
TRANSFORMERS_DEPLOY_MODELS = [
    "Qwen3.5-35B-A3B",
    "GLM-4.6V-Flash",
    "InternVL3.5-8B",
]
API_MODELS = ["gpt-5.1","gpt-5.2","gpt-5.4","gemini-3-pro","gemini-3-flash","gemini-3.1-pro","opus-4.6"]

# Load API keys and proxy settings from .env file
from dotenv import load_dotenv
load_dotenv(override=True)

if os.getenv("HTTP_PROXY"):
    os.environ["http_proxy"] = os.getenv("HTTP_PROXY")
if os.getenv("HTTPS_PROXY"):
    os.environ["https_proxy"] = os.getenv("HTTPS_PROXY")

def normalize_to_words(text):
    return re.findall(r'\b\w+\b', text)


def validate_output(original: str, generated: str) -> Tuple[bool, str]:
    """Validate the generated output format.

    合法输出只有两种：
    1. "NO HALLUCINATION"（不区分大小写，可带首尾空白）
    2. 去掉 <HALLUCINATION></HALLUCINATION> 标签后，单词序列与原文一致
       且标签配对正确、无嵌套
    """
    # 情况 1: 模型认为没有幻觉
    if generated.strip().upper() == "NO HALLUCINATION":
        return True, "Validation passed (NO HALLUCINATION)"

    # 情况 2: 带标签的输出，检查格式合法性
    # 2a. 标签配对
    open_tags = len(re.findall(r'<HALLUCINATION>', generated))
    close_tags = len(re.findall(r'</HALLUCINATION>', generated))
    if open_tags != close_tags:
        return False, f"Tags mismatch: {open_tags} vs {close_tags}"

    # 2b. 检查嵌套
    if re.search(r'<HALLUCINATION>(?:(?!</HALLUCINATION>).)*<HALLUCINATION>', generated):
        return False, "Nested tags detected"

    # 2c. 检查单词一致性
    cleaned_generated = re.sub(r'<HALLUCINATION>|</HALLUCINATION>', '', generated)
    original_words = normalize_to_words(original)
    generated_words = normalize_to_words(cleaned_generated)

    if original_words != generated_words:
        return False, "Word content does not match the original"

    return True, "Validation passed"

#Qwen3 think返回内容中，没有think开始的tag。
def post_process_think_output(generated_text: str) -> tuple[str, str]:
    """
    从模型输出中提取 think 和 result 内容。
    修改点：针对 <result> 标签，获取最后一个匹配项，防止 <think> 内部包含 <result> 导致提取错误。
    """
    if not generated_text:
        return "", ""

    # --- 1. 提取 Result (结果) ---
    # 使用 finditer 找到所有的 <result>...</result> 块
    # re.S (DOTALL) 允许 . 匹配换行符
    result_iter = list(re.finditer(r'<result>(.*?)</result>', generated_text, re.S))
    
    result_content = ""
    last_result_start_index = -1  # 用于后续辅助定位 think 的结束位置

    if result_iter:
        # 取最后一个匹配项 (解决 think 内嵌 result 的问题)
        last_match = result_iter[-1]
        result_content = last_match.group(1).strip()
        last_result_start_index = last_match.start()
    else:
        # 兜底策略：如果没找到 <result> 标签
        if "</think>" in generated_text:
            # 认为 </think> 之后的所有内容都是 result
            result_content = generated_text.split("</think>")[-1].strip()
        else:
            # 既无 result 标签也无 think 结束标签，返回空
            result_content = ""

    # --- 2. 提取 Think (思考) ---
    think_content = ""
    
    # 策略：以 </think> 为主要锚点
    if "</think>" in generated_text:
        # 取 </think> 之前的所有内容
        # 使用 rsplit 确保以最后一个 </think> 分割（防止极少数嵌套情况）
        raw_think = generated_text.rsplit("</think>", 1)[0].strip()
        
        # 清理开头的 <think>
        think_content = re.sub(r'^<think>\s*', '', raw_think, flags=re.IGNORECASE).strip()
    else:
        # 如果没有 </think> 结束标签 (可能是截断或格式错误)
        if last_result_start_index != -1:
             # 使用“最后一个 result 的开始位置”作为思考的结束点
             # 这样能避免把 think 里的 result 误当成有效内容，同时把前面的都归为 think
             think_content = generated_text[:last_result_start_index].strip()
             think_content = re.sub(r'^<think>\s*', '', think_content, flags=re.IGNORECASE).strip()
        else:
             # 全文都没有标签，视为全部是思考（或全部是Raw Text，视业务逻辑而定，这里保留原逻辑倾向于思考）
             think_content = generated_text.strip()
             think_content = re.sub(r'^<think>\s*', '', think_content, flags=re.IGNORECASE).strip()

    return result_content, think_content

def post_process_no_think_output(generated_text: str) -> str:
    """从模型输出中提取所需内容"""
    # 提取<result>标签内的内容
    result_match = re.search(r'<result>(.*?)</result>', generated_text, re.DOTALL)
    if result_match:
        return result_match.group(1).strip()
    else:
        return ""


API_MAX_LONG_SIDE = 1536  # API 模型统一将长边缩放到 1536px

def encode_image_to_base64(image_path, max_long_side=None):
    """将图片编码为 base64。若指定 max_long_side，则将长边缩放到该值再编码。"""
    if max_long_side is None:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    import io
    img = Image.open(image_path)
    w, h = img.size
    if max(w, h) > max_long_side:
        scale = max_long_side / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        logger.info(f"Resized {os.path.basename(image_path)}: {w}x{h} -> {new_w}x{new_h}")
    if img.mode in ("RGBA", "P", "LA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def prepare_inputs(items, image_dir, model_name, attempt_num=0, use_think=False):  #默认不使用think 
    """Prepare the input format for vLLM Chat"""
    inputs = []
    valid_indices = [] # record which item successfully loaded the image
    
    for idx, item in enumerate(items):
        # 从 item["image_path"] 中提取 ./test/ 之后的相对路径（含 category 子目录）
        # 例如 "./test/Movie/1CNEK7WA.jpg" -> "Movie/1CNEK7WA.jpg"
        raw_image_path = item.get("image_path", "")
        if "./test/" in raw_image_path:
            rel_path = raw_image_path.split("./test/", 1)[1]
        else:
            rel_path = item["id"]
        image_path = os.path.join(image_dir, rel_path)

        # try to load the image; if not found, search one level of subdirectories
        if not os.path.exists(image_path):
            filename = os.path.basename(rel_path)
            found = False
            for sub in os.listdir(image_dir):
                candidate = os.path.join(image_dir, sub, filename)
                if os.path.isfile(candidate):
                    image_path = candidate
                    found = True
                    break
            if not found:
                raise ValueError(f"Image not found {image_path}")
        
        try:
            max_side = API_MAX_LONG_SIDE if model_name in API_MODELS else None
            base64_image = encode_image_to_base64(image_path, max_long_side=max_side)
        except Exception as e:
            raise ValueError(f"Error encoding image {image_path} to base64: {e}")

        is_use_original = item["use_original_caption_flag"]
        input_caption = item["original_caption"] if is_use_original else item["hallucinated_caption"]

        # use the new generic function to generate prompt
        prompt_content = create_prompt(model_name, use_think, input_caption)

        # if retry, adjust the prompt to emphasize precision
        # if attempt_num > 0:
        #     prompt_content = prompt_content.replace("STRICT TASK:", "STRICT TASK (RETRY - BE PRECISE):")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/{'jpeg' if max_side else ('png' if image_path.lower().endswith('.png') else 'jpeg')};base64,{base64_image}"}
                    },
                    {"type": "text", "text": prompt_content}
                ]
            }
        ]
        inputs.append(messages)
        valid_indices.append(idx)
        
    return inputs, valid_indices


def call_model_batch(llm, model_name, batch_inputs, seed):
    """
    vLLM 批量推理接口。仅用于本地模型。
    """
    sampling_params = get_sampling_params(model_name, seed)
    print(f"SamplingParams configuration: {sampling_params}")
    print(f"Temperature: {sampling_params.temperature}")

    outputs = llm.chat(
        messages=batch_inputs,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    # Debug: inspect the first output object structure
    if outputs:
        first_out = outputs[0].outputs[0]
        print(f"[DEBUG] Output object type: {type(first_out)}")
        print(f"[DEBUG] Output object attrs: {[a for a in dir(first_out) if not a.startswith('_')]}")
        print(f"[DEBUG] .text repr (first 500): {repr(first_out.text[:500])}")
        if hasattr(first_out, 'reasoning_content'):
            rc = first_out.reasoning_content
            print(f"[DEBUG] .reasoning_content ({len(rc) if rc else 0} chars): {rc[:300] if rc else '(None)'}...")
        # 检查 token_ids
        if hasattr(first_out, 'token_ids'):
            print(f"[DEBUG] .token_ids (first 50): {first_out.token_ids[:50]}")

    results = []
    for output in outputs:
        out = output.outputs[0]
        # 优先使用 reasoning_content + text 拼接
        reasoning = getattr(out, 'reasoning_content', None) or ""
        text = out.text or ""
        if reasoning:
            combined = f"<think>{reasoning}</think>{text}"
        else:
            combined = text
        results.append(combined)
    return results


def _call_api_once(llm, model_name, messages, seed):
    """执行一次API调用（不含重试逻辑）。"""
    if model_name in ("gpt-5.1", "gpt-5.2", "gpt-5.4"):
        response = llm.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.01,
            seed=seed,
            max_completion_tokens=20480
        )
        return response.choices[0].message.content.strip()

    elif model_name in ("gemini-3-pro", "gemini-3-flash", "gemini-3.1-pro"):
        gemini_model_id = {
            "gemini-3-pro": "gemini-3-pro-preview",
            "gemini-3-flash": "gemini-3-flash-preview",
            "gemini-3.1-pro": "gemini-3.1-pro-preview",
        }[model_name]

        gemini_contents = []
        for msg in messages:
            for part in msg["content"]:
                if part["type"] == "image_url":
                    data_uri = part["image_url"]["url"]
                    header, b64_data = data_uri.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    image_bytes = base64.b64decode(b64_data)
                    gemini_contents.append(
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                    )
                elif part["type"] == "text":
                    gemini_contents.append(part["text"])

        response = llm.models.generate_content(
            model=gemini_model_id,
            contents=gemini_contents,
            config=types.GenerateContentConfig(
                temperature=0.01,
                max_output_tokens=20480,
                seed=seed,
            )
        )
        return response.text.strip() if response.text else ""

    elif model_name == "opus-4.6":
        # 将 OpenAI 格式的 messages 转为 Anthropic 格式
        anthropic_content = []
        for msg in messages:
            for part in msg["content"]:
                if part["type"] == "image_url":
                    data_uri = part["image_url"]["url"]
                    header, b64_data = data_uri.split(",", 1)
                    mime_type = header.split(":")[1].split(";")[0]
                    anthropic_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_data,
                        }
                    })
                elif part["type"] == "text":
                    anthropic_content.append({
                        "type": "text",
                        "text": part["text"]
                    })

        response = llm.messages.create(
            model="claude-opus-4-6",
            max_tokens=20480,
            temperature=0.01,
            messages=[{"role": "user", "content": anthropic_content}],
        )
        return response.content[0].text.strip()

    else:
        raise ValueError(f"未知的 API 模型名称: {model_name}")


# 指数退避重试间隔（秒）
API_RETRY_DELAYS = [1, 2, 4, 8]


def call_model_single(llm, model_name, messages, seed, item_id=None):
    """
    API 单条调用接口，带指数退避重试（1s, 2s, 4s, 8s）。
    """
    id_tag = f"[Item {item_id}] " if item_id else ""
    display_name = model_name
    if model_name in ("gemini-3-pro", "gemini-3-flash", "gemini-3.1-pro"):
        display_name = {"gemini-3-pro": "gemini-3-pro-preview", "gemini-3-flash": "gemini-3-flash-preview", "gemini-3.1-pro": "gemini-3.1-pro-preview"}[model_name]
    elif model_name == "opus-4.6":
        display_name = "claude-opus-4-6"

    logger.info(f"{id_tag}Calling {display_name} API (seed={seed})...")

    last_error = None
    for attempt_idx in range(1 + len(API_RETRY_DELAYS)):
        try:
            content = _call_api_once(llm, model_name, messages, seed)
            logger.info(f"{id_tag}{display_name} API returned {len(content)} chars")
            return content
        except Exception as e:
            last_error = e
            if attempt_idx < len(API_RETRY_DELAYS):
                delay = API_RETRY_DELAYS[attempt_idx]
                logger.warning(f"{id_tag}{display_name} API failed: {e}. Retrying in {delay}s... ({attempt_idx+1}/{len(API_RETRY_DELAYS)})")
                time.sleep(delay)
            else:
                logger.error(f"{id_tag}{display_name} API failed after {len(API_RETRY_DELAYS)} retries: {e}")

    raise last_error


def _load_image_internvl(image_path, input_size=448, max_num=12):
    """InternVL 专用图片预处理：dynamic_preprocess + ImageNet normalize"""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    image = Image.open(image_path).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def call_model_transformers(llm_tuple, model_name, image_path, prompt_text, seed):
    """
    Transformers 单条推理接口。用于 TRANSFORMERS_DEPLOY_MODELS 中的本地模型。
    """
    model, processor_or_tokenizer = llm_tuple

    if "InternVL" in model_name:
        # InternVL 专用：使用 model.chat() 接口
        tokenizer = processor_or_tokenizer
        pixel_values = _load_image_internvl(image_path, max_num=12).to(torch.bfloat16).cuda()
        question = '<image>\n' + prompt_text
        generate_kwargs = get_transformers_generate_kwargs(model_name, seed)
        response = model.chat(tokenizer, pixel_values, question, generate_kwargs)
        return response
    else:
        # Qwen / GLM 等标准 transformers 模型
        processor = processor_or_tokenizer
        qwen_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = processor.apply_chat_template(qwen_messages, tokenize=False, add_generation_prompt=True)
        image = Image.open(image_path)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)

        generate_kwargs = get_transformers_generate_kwargs(model_name, seed)

        with torch.no_grad():
            output_ids = model.generate(**inputs, **generate_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return output_text


def auto_resume(all_items, output_filename):
    """
    Resume from the existing results.
    Args:
        all_items: The list of items to process.
        output_filename: The filename to save the results.
    Returns:
        None. Save the results to the output file.
    """
    # resume from the existing results

    final_results = {}
    if os.path.exists(output_filename):

        logger.info(f"Found existing output file: {output_filename}")
        try:
            with open(output_filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # convert the list to dict (id -> item)，restore the final_results state
            # API 调用异常的条目（api_error=True）不加载，让它们被重新处理
            if isinstance(existing_data, list):
                api_error_count = 0
                for res in existing_data:
                    if "id" in res:
                        if res.get("api_error"):
                            api_error_count += 1
                            continue
                        final_results[res["id"]] = res
                logger.info(f"Loaded {len(final_results)} completed items from checkpoint")
                if api_error_count > 0:
                    logger.info(f"Skipping {api_error_count} API error records, will re-process them")
            else:
                logger.warning("Checkpoint file format is not List, restarting from scratch")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint (possibly corrupted or empty): {e}. Will rerun all data.")
            final_results = {}

    # calculate the pending items: remove the ids that are already in final_results from all_items
    processed_ids = set(final_results.keys())
    pending_items = [item for item in all_items if item["id"] not in processed_ids]

    if len(pending_items) == 0:
        logger.info("All data has been processed, skipping inference loop")
    else:
        logger.info(f"Remaining data to process: {len(pending_items)} / {len(all_items)}")

    return final_results, pending_items


def _save_results_to_disk(final_results, output_filename):
    """将 final_results 实时写入磁盘（主文件 + filtered 文件）。"""
    results_list = list(final_results.values())
    results_list.sort(key=lambda x: x["id"])

    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(results_list, f, indent=2, ensure_ascii=False)

    # 同步写入 filtered 文件
    filtered_results = [r for r in results_list if r.get("validation_passed") is True]
    if output_filename.endswith(".json"):
        filtered_filename = output_filename.replace(".json", "_filtered.json")
    else:
        filtered_filename = output_filename + "_filtered.json"

    with open(filtered_filename, 'w', encoding='utf-8') as f:
        json.dump(filtered_results, f, indent=2, ensure_ascii=False)


def _process_and_save_one(item, gen_text, attempt, seed, use_think, final_results, output_filename, max_attempts):
    """处理单条推理结果：后处理、验证、存入 final_results 并实时落盘。
    返回 True 表示通过或最终保存，False 表示需要重试。
    """
    # #####################print the raw generated text for debugging#####################
    # print(f"\n{'='*60}")
    # print(f"Processing item id={item['id']} (Seed {seed} | Attempt {attempt+1})")
    # print(f"[RAW OUTPUT] ({len(gen_text)} chars): {gen_text[:500]}{'...(truncated)' if len(gen_text) > 500 else ''}")

    if use_think:
        result_content, think_content = post_process_think_output(gen_text)
    else:
        result_content = post_process_no_think_output(gen_text)
        think_content = ""

    if not result_content:
        logger.warning(f"[Item {item['id']}] result_content is EMPTY!")
        has_think_open = "<think>" in gen_text.lower()
        has_think_close = "</think>" in gen_text.lower()
        has_result_open = "<result>" in gen_text
        has_result_close = "</result>" in gen_text
        logger.warning(f"[Item {item['id']}] TAG DETECTION: <think>={has_think_open}, </think>={has_think_close}, <result>={has_result_open}, </result>={has_result_close}")

    input_caption_used = item["original_caption"] if item["use_original_caption_flag"] else item["hallucinated_caption"]
    is_valid, msg = validate_output(input_caption_used, result_content)
    if not is_valid:
        logger.warning(f"[Item {item['id']}] Validation failed: {msg}")

    result = {
        "id": item["id"],
        "image_path": item["image_path"],
        "original_caption": item["original_caption"],
        "hallucinated_caption": item["hallucinated_caption"],
        "gt_hallucinated_caption_with_tags": item["hallucinated_caption_with_tags"],
        "hallucinated_caption_with_tags": result_content,
        "data_source": item.get("data_source", "empty"),
        "is_use_original": item["use_original_caption_flag"],
        "validation_passed": is_valid,
        "validation_message": msg if not is_valid else None,
        "attempt": attempt + 1,
        "seed": seed,
        "raw_output": gen_text
    }
    if use_think:
        result["thinking_process"] = think_content

    if is_valid or attempt == max_attempts - 1:
        final_results[item["id"]] = result
        _save_results_to_disk(final_results, output_filename)
        return True
    return False


def run_one_seed(llm, all_items, image_dir, seed, output_filename, max_attempts, model_name, use_think=False, api_concurrency=10):
    """
    Run the complete inference process for one seed (including retries).
    - vLLM 本地模型：批量推理，逐条保存
    - API 模型：并行调用（api_concurrency控制并发数），逐条保存（中断不丢失）
    """
    logger.info(f"Start running Seed: {seed}")
    logger.info(f"Target output file: {output_filename}")

    final_results, pending_items = auto_resume(all_items, output_filename)
    is_api_model = model_name in API_MODELS
    is_transformers_model = model_name in TRANSFORMERS_DEPLOY_MODELS

    for attempt in range(max_attempts):
        if len(pending_items) == 0:
            break

        logger.info(f"Seed {seed} | Attempt #{attempt+1} | Remaining: {len(pending_items)}")

        next_pending = []

        if is_transformers_model:
            # Transformers 本地模型：逐条推理、逐条保存
            logger.info(f"Transformers inference ({model_name}) processing {len(pending_items)} items...")
            for i in tqdm(range(len(pending_items)), desc=f"{model_name} Transformers"):
                item = pending_items[i]

                # 解析图片路径
                raw_image_path = item.get("image_path", "")
                if "./test/" in raw_image_path:
                    rel_path = raw_image_path.split("./test/", 1)[1]
                else:
                    rel_path = item["id"]
                abs_image_path = os.path.join(image_dir, rel_path)

                is_use_original = item["use_original_caption_flag"]
                input_caption = item["original_caption"] if is_use_original else item["hallucinated_caption"]
                prompt_text = create_prompt(model_name, use_think, input_caption)

                try:
                    gen_text = call_model_transformers(llm, model_name, abs_image_path, prompt_text, seed)
                    # #####################print the raw generated text for debugging#####################
                    # print(f"gen_text (Seed {seed} | Attempt {attempt+1} | id={item['id']}): {gen_text[:]}...")  
                except Exception as e:
                    print(f"\nTransformers 推理失败 (第 {i} 条, id={item['id']}): {e}")
                    if attempt == max_attempts - 1:
                        error_result = {
                            "id": item["id"],
                            "image_path": item["image_path"],
                            "original_caption": item["original_caption"],
                            "hallucinated_caption": item["hallucinated_caption"],
                            "gt_hallucinated_caption_with_tags": item["hallucinated_caption_with_tags"],
                            "hallucinated_caption_with_tags": "",
                            "data_source": item.get("data_source", "empty"),
                            "is_use_original": item["use_original_caption_flag"],
                            "validation_passed": False,
                            "validation_message": f"Transformers inference failed: {e}",
                            "attempt": attempt + 1,
                            "seed": seed,
                            "raw_output": ""
                        }
                        final_results[item["id"]] = error_result
                        _save_results_to_disk(final_results, output_filename)
                    else:
                        next_pending.append(item)
                    continue

                passed = _process_and_save_one(
                    item, gen_text, attempt, seed, use_think,
                    final_results, output_filename, max_attempts
                )
                if not passed:
                    next_pending.append(item)

        elif is_api_model:
            batch_inputs, valid_indices = prepare_inputs(pending_items, image_dir, model_name, attempt, use_think)

            # API 模型：并行调用、逐条处理、逐条保存
            # API 调用异常和格式验证失败都消耗 attempt 次数
            # 区别：API 异常标记 api_error=True，再次运行时会被 auto_resume 重新处理
            logger.info(f"Calling API ({model_name}) for {len(batch_inputs)} items with concurrency={api_concurrency}...")

            def _call_api(i):
                original_idx = valid_indices[i]
                item = pending_items[original_idx]
                try:
                    gen_text = call_model_single(llm, model_name, batch_inputs[i], seed, item_id=item["id"])
                    return i, gen_text, None
                except Exception as e:
                    logger.warning(f"API call failed (item {i}, id={item['id']}): {e}")
                    return i, None, e

            with concurrent.futures.ThreadPoolExecutor(max_workers=api_concurrency) as executor:
                future_to_idx = {executor.submit(_call_api, i): i for i in range(len(batch_inputs))}
                for future in tqdm(concurrent.futures.as_completed(future_to_idx), total=len(future_to_idx), desc=f"{model_name} API"):
                    i, gen_text, error = future.result()
                    original_idx = valid_indices[i]
                    item = pending_items[original_idx]

                    if error:
                        api_error_result = {
                            "id": item["id"],
                            "image_path": item["image_path"],
                            "original_caption": item["original_caption"],
                            "hallucinated_caption": item["hallucinated_caption"],
                            "gt_hallucinated_caption_with_tags": item["hallucinated_caption_with_tags"],
                            "hallucinated_caption_with_tags": "",
                            "data_source": item.get("data_source", "empty"),
                            "is_use_original": item["use_original_caption_flag"],
                            "validation_passed": False,
                            "validation_message": f"API call failed: {error}",
                            "api_error": True,
                            "attempt": attempt + 1,
                            "seed": seed,
                            "raw_output": ""
                        }
                        if attempt == max_attempts - 1:
                            final_results[item["id"]] = api_error_result
                            _save_results_to_disk(final_results, output_filename)
                        else:
                            next_pending.append(item)
                        continue

                    # API 调用成功：走正常的验证+重试流程
                    passed = _process_and_save_one(
                        item, gen_text, attempt, seed, use_think,
                        final_results, output_filename, max_attempts
                    )
                    if not passed:
                        next_pending.append(item)
        else:
            batch_inputs, valid_indices = prepare_inputs(pending_items, image_dir, model_name, attempt, use_think)
            print(f"[vLLM] Prepared {len(batch_inputs)} inputs, valid_indices count: {len(valid_indices)}")
            if len(batch_inputs) > 0:
                print(f"[vLLM] First input prompt (last 200 chars): ...{str(batch_inputs[0])[-200:]}")

            # vLLM 本地模型：批量推理
            outputs_text_list = call_model_batch(llm, model_name, batch_inputs, seed)
            print(f"[vLLM] Got {len(outputs_text_list)} outputs")
            if len(outputs_text_list) > 0:
                print(f"[vLLM] First output ({len(outputs_text_list[0])} chars): {outputs_text_list[0][:200]}...")

            for i, gen_text in enumerate(outputs_text_list):
                original_idx = valid_indices[i]
                item = pending_items[original_idx]

                passed = _process_and_save_one(
                    item, gen_text, attempt, seed, use_think,
                    final_results, output_filename, max_attempts
                )
                if not passed:
                    next_pending.append(item)

        pending_items = next_pending
        logger.info(f"[ATTEMPT SUMMARY] Seed {seed} | Attempt #{attempt+1} done: "
                    f"processed={len(final_results)}, pending_next={len(pending_items)}")

    # 最终保存一次，确保完整
    _save_results_to_disk(final_results, output_filename)

    results_list = list(final_results.values())
    pass_count = sum(1 for r in results_list if r["validation_passed"])
    logger.info(f"Seed {seed} completed! Results saved to {output_filename}")
    logger.info(f"Statistics: Total {len(results_list)}, Passed {pass_count}, Failed/Retried {len(results_list) - pass_count}")

    if output_filename.endswith(".json"):
        filtered_filename = output_filename.replace(".json", "_filtered.json")
    else:
        filtered_filename = output_filename + "_filtered.json"
    filtered_count = sum(1 for r in results_list if r.get("validation_passed") is True)
    logger.info(f"Filtered results saved to: {filtered_filename} (Total {filtered_count} items)")
    



def load_eval_input_data(input_dir, limit_per_category=None):
    """
    Load evaluation input data from a directory.
    Uses the is_modified field from each item to determine whether to use
    the original caption (is_modified=False) or hallucinated caption (is_modified=True).
    Args:
        input_dir: The directory containing the evaluation input data.
        limit_per_category: If set, only load the first N items per category (based on image_path subdirectory).
    Returns:
        A list of input data items.
    """
    logger.info(f"Loading evaluation input data from {input_dir}...")
    if os.path.exists(input_dir):
        all_items = []
        files = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])
        for f in files:
            with open(os.path.join(input_dir, f), 'r') as fp:
                data = json.load(fp)
                if isinstance(data, dict): data = [data]
                all_items.extend(data)
        logger.info(f"Loaded {len(all_items)} test data items")

        # 按类别截取前 limit 条
        if limit_per_category is not None and limit_per_category > 0:
            from collections import defaultdict
            category_counts = defaultdict(int)
            filtered_items = []
            for item in all_items:
                # 从 image_path 提取类别子目录，如 "./test/Movie/xxx.jpg" -> "Movie"
                parts = item.get("image_path", "").split("/")
                cat = parts[2] if len(parts) > 2 else "unknown"
                if category_counts[cat] < limit_per_category:
                    filtered_items.append(item)
                    category_counts[cat] += 1
            logger.info(f"Limit per category: {limit_per_category}, kept {len(filtered_items)} items")
            for cat, cnt in sorted(category_counts.items()):
                logger.info(f"  {cat}: {cnt}")
            all_items = filtered_items

        # 根据 is_modified 字段决定使用 original 还是 hallucinated caption
        num_original = 0
        for item in all_items:
            is_modified = item.get("is_modified", True)
            item["use_original_caption_flag"] = not is_modified
            if not is_modified:
                num_original += 1

        logger.info(f"Using {num_original} original captions, {len(all_items)-num_original} hallucinated captions")

        return all_items
    else:
        raise FileNotFoundError(f"Input directory not found: {input_dir}")


def create_llm(model_path, model_select):
    """
    Create a LLM object based on the model selection.
    Args:
        model_path: The path to the model.
        model_select: The model selection.
    Returns:
        A LLM object.
    """
    if model_select in VLLM_DEPLOY_MODELS:
        logger.info(f"Using vLLM to deploy local model: {model_path}")
        llm = LLM(
            model=model_path,
            trust_remote_code=True,
            max_model_len=32768,
            limit_mm_per_prompt={"image": 1}
        )
        
    elif model_select in TRANSFORMERS_DEPLOY_MODELS:
        logger.info(f"Using Transformers to deploy local model: {model_path}")
        if "InternVL" in model_select:
            # InternVL 使用自定义 model.chat() 接口，需要 AutoModel + AutoTokenizer
            from transformers import AutoTokenizer, AutoModel
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
                device_map="auto",
            ).eval()
            llm = (model, tokenizer)
        else:
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            model.eval()
            llm = (model, processor)

    elif model_select in API_MODELS:
        if model_select in ("gpt-5.1", "gpt-5.2", "gpt-5.4"):
            assert os.getenv("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL")
            client_kwargs = {"api_key": api_key, "max_retries": 0, "timeout": 300}
            if base_url:
                client_kwargs["base_url"] = base_url
            llm = OpenAI(**client_kwargs)
        elif model_select in ("gemini-3-pro", "gemini-3-flash", "gemini-3.1-pro"):
            assert os.getenv("GEMINI_API_KEY") is not None, "GEMINI_API_KEY is not set"
            api_key = os.getenv("GEMINI_API_KEY")
            base_url = os.getenv("GEMINI_BASE_URL")
            client_kwargs = {"api_key": api_key}
            if base_url:
                client_kwargs["http_options"] = {"base_url": base_url}
            llm = genai.Client(**client_kwargs)
        elif model_select == "opus-4.6":
            assert os.getenv("ANTHROPIC_API_KEY") is not None, "ANTHROPIC_API_KEY is not set"
            api_key = os.getenv("ANTHROPIC_API_KEY")
            base_url = os.getenv("ANTHROPIC_BASE_URL")
            client_kwargs = {"api_key": api_key, "max_retries": 0, "timeout": 300}
            if base_url:
                client_kwargs["base_url"] = base_url
            llm = anthropic.Anthropic(**client_kwargs)

        logger.info(f"Using API model: {model_select}")
        
    else:
        raise ValueError(f"Unknown model: {model_select}\n Available models:\nvllm deploy models: {VLLM_DEPLOY_MODELS}\ntransformers deploy models: {TRANSFORMERS_DEPLOY_MODELS}\napi models: {API_MODELS}")

    return llm

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL vLLM Inference with Multi-Seed")
    parser.add_argument("--model_path", help="Huggingface model id or local path")
    parser.add_argument("--image_dir", required=True, help="Image directory")
    parser.add_argument("--input_dir", required=True, help="JSON input directory")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--use_think",action='store_true', help="Whether to use the CoT prompt")  
    parser.add_argument("--seeds", required=True, nargs='+', type=int, help="Seed list")
    parser.add_argument("--model_select",required=True, help="Model selection")
    parser.add_argument("--max_attempts", type=int, default=2, help="Maximum number of attempts")
    parser.add_argument("--limit", type=int, default=None, help="Only run first N items per category (for quick testing)")
    parser.add_argument("--api_concurrency", type=int, default=10, help="Number of parallel API calls (default: 10)")
    args = parser.parse_args()

    # check if output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.model_select in PROMPT_TEMPLATES, f"Model {args.model_select} not found in PROMPT_TEMPLATES"

    all_items = load_eval_input_data(args.input_dir, limit_per_category=args.limit)

    llm = create_llm(args.model_path, args.model_select)
    
    if args.model_select in API_MODELS:
        logger.info(f"API concurrency: {args.api_concurrency}")

    for seed in args.seeds:
        filename = f"tested_model_output_seed_{seed}.json"
        run_one_seed(
            llm=llm,
            all_items=all_items,
            image_dir=args.image_dir,
            seed=seed,
            output_filename=os.path.join(args.output_dir, filename),
            max_attempts=args.max_attempts,
            model_name=args.model_select,
            use_think=args.use_think,
            api_concurrency=args.api_concurrency)

    print("\nAll seeds have been executed")

if __name__ == "__main__":
    main()