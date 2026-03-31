#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization script for Hallucination Attention Heatmaps & Object Grounding.
Compatible with Qwen2.5-VL, Qwen3-VL.
Updates:
1. Pre-filter images.
2. Refactored --all_word_vis: Visualizes predefined objects in the prompt "Is there a {obj} in the image?".
"""

import os
import json
import argparse
import torch
import numpy as np
import cv2
import re
import gc
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 预定义配置 =================

# 预定义的15个常见物体
TARGET_OBJECTS = [
    "plane", "train", "bread", "fork", "cat", 
    "dog", "car", "person", "flower", "book", 
    "chair", "table", "bird", "apple"
]

# ================= 工具函数 =================

def create_prompt(prompt_data, model_name, use_think, input_caption):
    """原模式使用的 Prompt 构造"""
    mode_key = "think" if use_think else "no_think"
    if model_name not in prompt_data:
        if "Qwen2.5-VL-7B" in prompt_data:
            template = prompt_data["Qwen2.5-VL-7B"][mode_key]
        else:
            raise ValueError(f"Error: Model '{model_name}' configuration not found")
    else:
        template = prompt_data[model_name][mode_key]
    return template.format(caption=input_caption)

def get_valid_result_span(generated_text: str, use_think: bool) -> tuple[int, int]:
    """原模式使用的结果截取"""
    full_len = len(generated_text)
    if use_think:
        result_iter = list(re.finditer(r'<result>(.*?)</result>', generated_text, re.S))
        if result_iter: return result_iter[-1].span(1)
        if "</think>" in generated_text:
            split_token = "</think>"
            return generated_text.rfind(split_token) + len(split_token), full_len
        return -1, -1
    else:
        result_match = re.search(r'<result>(.*?)</result>', generated_text, re.S)
        if result_match: return result_match.span(1)
        if "<HALLUCINATION>" in generated_text: return 0, full_len
        return -1, -1

def get_tokens_from_char_span(offsets, abs_start, abs_end, max_len):
    """根据字符绝对位置，找对应的 Token Indices"""
    indices = []
    for i, (o_start, o_end) in enumerate(offsets):
        if max(abs_start, o_start) < min(abs_end, o_end):
            indices.append(i)
    return [x for x in indices if x < max_len]

def compute_heatmap_for_tokens(outputs, current_token_indices, image_token_indices, start_layer_idx):
    """通用 Attention 计算逻辑 - 返回每层独立的 attention map"""
    layer_maps = {}
    for i in range(start_layer_idx, len(outputs.attentions)):
        layer_attn = outputs.attentions[i][0] # [heads, seq_len, seq_len]
        # 切片: [heads, text_tokens, image_tokens]
        attn_slice = layer_attn[:, current_token_indices, :][:, :, image_token_indices]
        attn_slice = attn_slice.float()
        layer_map = attn_slice.max(dim=0)[0].mean(dim=0)
        layer_maps[i] = layer_map

    return layer_maps if layer_maps else None

def process_heatmap_visual(attn_map, image_size, inputs):
    """通用热力图后处理"""
    if "image_grid_thw" in inputs:
        thw = inputs["image_grid_thw"]
        if torch.is_tensor(thw): thw = thw.cpu().numpy()
        grid_h, grid_w = thw[0][1], thw[0][2]
    else:
        side = int(np.sqrt(len(attn_map)))
        grid_h, grid_w = side, side

    if len(attn_map) != grid_h * grid_w:
        side = int(np.sqrt(len(attn_map)))
        attn_map = attn_map[:side*side].reshape(side, side)
        attn_map = cv2.resize(attn_map, (grid_w, grid_h))
    else:
        attn_map = attn_map.reshape(grid_h, grid_w)
        
    heatmap = cv2.resize(attn_map, image_size)
    
    heatmap = cv2.GaussianBlur(heatmap, (35, 35), 0)
    q_min, q_max = np.percentile(heatmap, 5), np.percentile(heatmap, 99.5)
    heatmap = np.clip(heatmap, q_min, q_max)
    delta = q_max - q_min
    heatmap_norm = (heatmap - q_min) / delta if delta > 1e-9 else np.zeros_like(heatmap)
    
    return heatmap_norm

# ================= 新增逻辑：指定物体 Query 可视化 =================
def get_object_query_heatmap(model, processor, image_path, object_name, device):
    """
    全词测试模式专用：
    1. 构造 "Is there a {object}?"
    2. 模型生成回复 (Generate)。
    3. 获取回复的每个词的热力图 (All Word Vis)。
    4. 获取 Prompt 中 {object} 的热力图。
    """
    # 1. Image Load
    try:
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"  [Error] Image open failed: {e}")
        return None, None, None

    # 2. 第一次构造输入：用于 Generate
    prompt_text = f'Is there a {object_name} in the image? Only answer "Yes" or "No".'
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}
    ]

    try:
        # 预处理
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        )
        # 转换为普通 dict 并移动到 device
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in dict(inputs).items()}
        
        # 修正：类型对齐（Qwen系列常见问题）
        if "pixel_values" in inputs and inputs["pixel_values"].dtype == torch.float32:
             inputs["pixel_values"] = inputs["pixel_values"].to(dtype=model.dtype)

        # === 3. Generate (获取模型回复) ===
        # 限制 max_new_tokens 防止输出太长爆显存
        generated_ids = model.generate(**inputs, max_new_tokens=50, use_cache=True)
        
        # 提取生成的 tokens (去除 input 部分)
        # ⭐ 错误修复点：inputs 是 dict，必须用 ["input_ids"] 访问，不能用 .input_ids
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    except Exception as e:
        print(f"  [Error] Generation failed: {e}")
        # 如果生成失败，清理显存并返回
        del inputs
        return None, None, None

    # === 4. 第二次构造输入：用于 Forward (User + Assistant) ===
    # 将模型刚刚生成的 output_text 拼回去，计算注意力
    messages_full = messages + [{"role": "assistant", "content": output_text}]
    
    try:
        text_full = processor.apply_chat_template(messages_full, tokenize=False, add_generation_prompt=False)
        # 重新处理输入，这次带上了 offsets 用于定位
        inputs_full = processor(
            text=[text_full], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", return_offsets_mapping=True
        )
        offset_mapping = inputs_full.pop("offset_mapping")[0]
        
        inputs_fwd = {}
        for k, v in dict(inputs_full).items():
            if torch.is_tensor(v):
                if "pixel" in k and v.dtype == torch.float32: v = v.to(dtype=model.dtype)
                inputs_fwd[k] = v.to(device)
            elif isinstance(v, list):
                inputs_fwd[k] = [x.to(device) if torch.is_tensor(x) else x for x in v]
            else: inputs_fwd[k] = v

        # === 5. Forward Pass ===
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs_fwd, output_attentions=True)

    except Exception as e:
        print(f"  [Error] Forward with generation failed: {e}")
        return None, None, None

    # === 6. 解析 Indices ===
    input_ids = inputs_fwd["input_ids"][0].cpu()
    full_decoded_text = processor.tokenizer.decode(input_ids)
    
    # 6.1 找 Vision Range
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    try:
        vision_start_idx = (input_ids == vision_start_id).nonzero(as_tuple=True)[0][0].item()
        vision_end_idx = (input_ids == vision_end_id).nonzero(as_tuple=True)[0][0].item()
        image_token_indices = list(range(vision_start_idx + 1, vision_end_idx))
    except: 
        print("  [Error] Vision tokens not found")
        return None, None, None

    results_list = []
    # 如果显存不够，可以把层数调小，比如只看最后5层
    start_layer_idx = max(0, len(outputs.attentions) - 20)

    # === 任务 A: 可视化 Prompt 中的 Object (Is there a {object}...) ===
    # 策略：在 output_text 出现之前的位置找 object_name
    gen_start_pos = full_decoded_text.rfind(output_text) if output_text else len(full_decoded_text)
    prompt_part = full_decoded_text[:gen_start_pos]
    
    # 注意：rfind 可能会找到 template 里的词，但 object 通常只出现在 user query 里
    obj_start = prompt_part.rfind(object_name)
    if obj_start != -1:
        obj_end = obj_start + len(object_name)
        obj_tokens = get_tokens_from_char_span(offset_mapping, obj_start, obj_end, len(input_ids))
        
        if obj_tokens:
            layer_maps = compute_heatmap_for_tokens(outputs, obj_tokens, image_token_indices, start_layer_idx)
            if layer_maps is not None:
                layer_heatmaps = {li: process_heatmap_visual(lm.cpu().numpy(), image.size, inputs_fwd) for li, lm in layer_maps.items()}
                results_list.append({
                    "type": "prompt_object",
                    "phrase": object_name,
                    "layer_heatmaps": layer_heatmaps,
                    "token_indices": obj_tokens
                })
    else:
        # 这是一个常见的情况：分词器可能把单词前面加了空格，或者把单词拆碎了
        # 简单重试：尝试找 " " + object_name
        obj_start = prompt_part.rfind(" " + object_name)
        if obj_start != -1:
             obj_end = obj_start + len(" " + object_name)
             obj_tokens = get_tokens_from_char_span(offset_mapping, obj_start, obj_end, len(input_ids))
             if obj_tokens:
                layer_maps = compute_heatmap_for_tokens(outputs, obj_tokens, image_token_indices, start_layer_idx)
                if layer_maps is not None:
                    layer_heatmaps = {li: process_heatmap_visual(lm.cpu().numpy(), image.size, inputs_fwd) for li, lm in layer_maps.items()}
                    results_list.append({
                        "type": "prompt_object",
                        "phrase": object_name,
                        "layer_heatmaps": layer_heatmaps,
                        "token_indices": obj_tokens
                    })

    # === 任务 B: 可视化 Generated Text 中的每一个 Word ===
    if output_text:
        gen_abs_start = gen_start_pos
        gen_abs_end = len(full_decoded_text)
        
        for idx, (o_start, o_end) in enumerate(offset_mapping):
            if idx >= len(input_ids): break
            
            # 必须在生成内容的范围内
            if o_start >= gen_abs_start and o_end <= gen_abs_end:
                token_str = processor.tokenizer.decode([input_ids[idx]])
                if not token_str.strip(): continue
                
                curr_indices = [idx]
                layer_maps = compute_heatmap_for_tokens(outputs, curr_indices, image_token_indices, start_layer_idx)

                if layer_maps is not None:
                    layer_heatmaps = {li: process_heatmap_visual(lm.cpu().numpy(), image.size, inputs_fwd) for li, lm in layer_maps.items()}
                    results_list.append({
                        "type": "gen_word",
                        "phrase": token_str,
                        "layer_heatmaps": layer_heatmaps,
                        "token_indices": curr_indices
                    })

    # 清理显存
    del outputs, inputs_fwd
    if 'inputs' in locals(): del inputs
    
    return results_list, image, output_text

# ================= 原有逻辑：幻觉检测 =================
def get_multi_attention_heatmaps(model, processor, image_path, prompt_text, generated_text, device, use_think):
    # ================= 1. 图像加载与预处理 =================
    try:
        image = Image.open(image_path).convert("RGB")
        # 强制缩小图片防止显存爆炸 (OOM)
        max_dimension = 1024
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    except Exception as e:
        print(f"  [Error] Image open failed: {e}")
        return None, None

    # ================= 2. 构造模型输入 =================
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": prompt_text}
        ]},
        {"role": "assistant", "content": generated_text}
    ]

    try:
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # 获取原始 BatchFeature
        raw_inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        
        # === 鲁棒的数据搬运逻辑 (处理 List/Tensor 混合结构) ===
        inputs = {}
        raw_dict = dict(raw_inputs) # 强转 dict，避免 BatchFeature 的锅
        
        for k, v in raw_dict.items():
            if torch.is_tensor(v):
                # 像素值类型对齐 (fp32 -> fp16/bf16)
                if "pixel" in k and v.dtype == torch.float32:
                     v = v.to(dtype=model.dtype)
                inputs[k] = v.to(device)
            elif isinstance(v, list):
                new_list = []
                for item in v:
                    if torch.is_tensor(item):
                        if item.dtype == torch.float32:
                             item = item.to(dtype=model.dtype)
                        new_list.append(item.to(device))
                    else:
                        new_list.append(item)
                inputs[k] = new_list
            else:
                inputs[k] = v
        # ====================================================

    except Exception as e:
        print(f"  [Error] Input preprocessing failed: {e}")
        return None, None

    # ================= 3. 前向传播 (Forward Pass) =================
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
    except Exception as e:
        print(f"  [Error] Model forward failed: {e}")
        return None, None

    # ================= 4. 定位视觉 Token (Vision Indices) =================
    # 为了后续 numpy 操作，先把 input_ids 转回 CPU
    if isinstance(inputs["input_ids"], list):
        input_ids = inputs["input_ids"][0].cpu()
    else:
        input_ids = inputs["input_ids"][0].cpu()
    
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    try:
        vision_start_idx = (input_ids == vision_start_id).nonzero(as_tuple=True)[0][0].item()
        vision_end_idx = (input_ids == vision_end_id).nonzero(as_tuple=True)[0][0].item()
    except IndexError:
        # Qwen 有时只有 pad，没有 vision token (如果图太小或被截断)
        return None, None
        
    image_token_indices = list(range(vision_start_idx + 1, vision_end_idx))
    
    # ================= 5. 定位文本范围与幻觉标签 =================
    # 获取有效生成内容（去除 CoT/Think 部分）
    valid_start_local, valid_end_local = get_valid_result_span(generated_text, use_think)
    if valid_start_local == -1: return None, None

    valid_content = generated_text[valid_start_local:valid_end_local]
    
    # === 修改点：使用分组正则，同时捕获标签本身和内容 ===
    # Group 1: <HALLUCINATION>
    # Group 2: 内容
    # Group 3: </HALLUCINATION>
    pattern = re.compile(r"(<HALLUCINATION>)(.*?)(</HALLUCINATION>)", re.DOTALL)
    matches = list(pattern.finditer(valid_content))
    
    if not matches: return None, None

    # 计算全局偏移量
    full_text = processor.tokenizer.decode(input_ids)
    gen_start_offset = full_text.rfind(generated_text)
    if gen_start_offset == -1:
        gen_start_offset = full_text.rfind(generated_text[:50]) # 模糊匹配兜底
        if gen_start_offset == -1: return None, None

    # 获取所有 Token 的字符偏移映射
    enc = processor.tokenizer(full_text, return_offsets_mapping=True, return_tensors="pt")
    offsets = enc.offset_mapping[0] # (num_tokens, 2)

    # 准备聚合参数
    num_layers_to_check = 20
    start_layer_idx = max(0, len(outputs.attentions) - num_layers_to_check)
    
    results = []

    # ================= 6. 循环处理每个匹配项 (Tags + Content) =================
    for match in matches:
        # 定义要提取的三部分：开始标签、内容、结束标签
        groups = [
            (match.group(1), match.span(1), "tag_open"),  # <HALLUCINATION>
            (match.group(2), match.span(2), "content"),   # 具体幻觉词
            (match.group(3), match.span(3), "tag_close")  # </HALLUCINATION>
        ]
        
        for phrase, (rs, re_pos), type_key in groups:
            if not phrase: continue # 防止空匹配
            
            # 计算这一小段文本在 full_text 中的绝对位置
            abs_start = gen_start_offset + valid_start_local + rs
            abs_end = gen_start_offset + valid_start_local + re_pos
            
            # 映射到 Token ID
            current_token_indices = []
            for i, (o_start, o_end) in enumerate(offsets):
                # 简单的区间交集判断
                if max(abs_start, o_start) < min(abs_end, o_end):
                    current_token_indices.append(i)
            
            # 过滤无效 Token
            current_token_indices = [x for x in current_token_indices if x < len(input_ids)]
            if not current_token_indices: continue

            # === 计算每层独立的 Attention Map ===
            layer_heatmaps = {}
            for i in range(start_layer_idx, len(outputs.attentions)):
                layer_attn = outputs.attentions[i][0] # GPU Tensor

                # 切片: [Heads, Target_Tokens, Image_Tokens]
                attn_slice = layer_attn[:, current_token_indices, :][:, :, image_token_indices]
                attn_slice = attn_slice.float()

                # 聚合: Max over Heads -> Mean over Targets
                layer_map = attn_slice.max(dim=0)[0].mean(dim=0)

                hm = process_heatmap_visual(layer_map.cpu().numpy(), image.size, inputs)
                layer_heatmaps[i] = hm

            if not layer_heatmaps: continue

            # 设置显示名称 (区分是内容还是标签)
            display_phrase = phrase if type_key == "content" else f"[TAG] {phrase}"

            results.append({
                "phrase": display_phrase,
                "layer_heatmaps": layer_heatmaps,
                "token_indices": current_token_indices,
                "type": type_key
            })

    # 清理显存
    del outputs
    del inputs
    del raw_inputs
    torch.cuda.empty_cache()

    return results, image

def save_visualization(image, heatmap, output_path):
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    heatmap_cleaned = heatmap.copy()
    heatmap_cleaned[heatmap_cleaned < 0.05] = 0
    heatmap_uint8 = np.uint8(255 * heatmap_cleaned)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    mask = np.power(heatmap_cleaned, 0.6) 
    mask = np.expand_dims(mask, axis=2)
    overlay = (image_bgr * (1 - mask) + heatmap_color * mask).astype(np.uint8)
    cv2.imwrite(output_path, overlay)

# ================= Main =================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_file", default="prompt.json")
    parser.add_argument("--model_select", required=True)
    parser.add_argument("--use_think", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--all_word_vis", action="store_true") # 此时这代表 Object Query + All Gen Word Vis

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # === 关键修改 1: 强制 Batch Size 为 1 以解决 OOM ===
    if args.all_word_vis:
        print("⚠️ All Word Vis mode detected: Forcing batch_size=1 to prevent OOM.")
        args.batch_size = 1

    # 1. Load & Filter Data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    if args.max_samples: raw_data = raw_data[:args.max_samples]
    
    valid_data = []
    print("🔍 Pre-filtering images...")
    for item in raw_data:
        img_name = item.get("image_path", "") or item["id"]
        if os.path.isabs(img_name):
            full_path = img_name
        else:
            # 先尝试保留子目录结构（如 ./test/Movie/xxx.jpg -> image_dir/Movie/xxx.jpg）
            # 去掉开头的 ./test/ 或 test/ 前缀，保留子目录
            rel = img_name.lstrip("./")
            if rel.startswith("test/"):
                rel = rel[len("test/"):]
            full_path = os.path.join(args.image_dir, rel)
            # 兜底：直接用 basename
            if not os.path.exists(full_path):
                full_path = os.path.join(args.image_dir, os.path.basename(img_name))
        if os.path.exists(full_path):
            item["_full_image_path"] = full_path
            valid_data.append(item)
            
    total_samples = len(valid_data)
    print(f"📂 Valid records: {total_samples}")

    prompt_data = {}
    if not args.all_word_vis:
        if os.path.exists(args.prompt_file):
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_data = json.load(f)
        else:
            print(f"⚠️ Prompt file '{args.prompt_file}' not found, skipping prompt_data loading.")

    vis_metadata = []
    num_batches = (total_samples + args.batch_size - 1) // args.batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min((batch_idx + 1) * args.batch_size, total_samples)
        current_batch = valid_data[start_idx:end_idx]
        
        print(f"\n🔄 [Batch {batch_idx+1}/{num_batches}] Processing {start_idx} to {end_idx}...")
        
        # 每次循环（如果是all_word_vis模式，就是每张图）都重新加载模型
        try:
            model = AutoModelForImageTextToText.from_pretrained(
                args.model_path, torch_dtype=torch.float16, device_map={"": device}, 
                trust_remote_code=True, attn_implementation="eager"
            )
            processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
        except Exception as e:
            print(f"❌ Model load error: {e}")
            break
            
        for i, item in enumerate(current_batch):
            global_idx = start_idx + i
            img_path = item["_full_image_path"]
            safe_id = item['id'].replace("/", "_").replace(".", "_")
            print(f"[{global_idx+1}/{total_samples}] Processing {item['id']}...")

            # === 分支 A: All Word Vis (Object Query + Gen Response) ===
            if args.all_word_vis:
                entry = {
                    "id": item["id"], 
                    "original_image": img_path, 
                    "mode": "object_query_plus_gen",
                    "objects_data": [] # 存放15个物体的详细结果
                }
                
                for obj_name in TARGET_OBJECTS:
                    try:
                        # 调用修改后的函数，接收 3 个返回值
                        results_list, image_obj, gen_text = get_object_query_heatmap(
                            model, processor, img_path, obj_name, device
                        )
                        
                        if results_list:
                            obj_entry = {
                                "object_query": obj_name,
                                "generated_response": gen_text, # 保存模型输出
                                "vis_items": []
                            }
                            
                            for h_idx, res in enumerate(results_list):
                                phrase = res["phrase"]
                                p_type = res["type"] # 'prompt_object' or 'gen_word'
                                safe_phrase = re.sub(r'[^\w\s-]', '', phrase).strip().replace(" ", "")[:15]

                                # 每层单独保存一张图
                                layer_paths = {}
                                for layer_idx, hm in res["layer_heatmaps"].items():
                                    # 文件名：ID_ObjName_Idx_Type_Word_L{layer}.jpg
                                    img_filename = f"{safe_id}_{obj_name}_{h_idx:03d}_{p_type}_{safe_phrase}_L{layer_idx:02d}.jpg"
                                    save_path = os.path.join(args.output_dir, img_filename)
                                    save_visualization(image_obj, hm, save_path)
                                    layer_paths[layer_idx] = img_filename

                                obj_entry["vis_items"].append({
                                    "type": p_type,
                                    "phrase": phrase,
                                    "layer_heatmap_paths": layer_paths
                                })
                            
                            entry["objects_data"].append(obj_entry)
                            
                    except Exception as e:
                        print(f"   Error processing object '{obj_name}': {e}")
                
                vis_metadata.append(entry)

            # === 分支 B: Normal Mode (Hallucination Detection) ===
            else:
                # ... (保持原有的 Normal Mode 逻辑不变) ...
                if "<HALLUCINATION>" not in item["gen_text"]: continue
                # (此处代码省略，与上一版一致)
                pass # 占位，实际运行时请保留原有代码

        # Batch cleanup
        meta_path = os.path.join(args.output_dir, "visualization_metadata.json")
        # 增量/覆盖保存 JSON
        # 注意：如果数据量极大，建议追加写入模式，但 JSON 结构不支持直接追加，
        # 这里还是采用每次 Batch 完重写整个 list 的方式（适合几千条以内）
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(vis_metadata, f, indent=2, ensure_ascii=False)
        
        print("🧹 Cleaning VRAM...")
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n✅ All Done.")

if __name__ == "__main__":
    main()