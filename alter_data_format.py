#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 1: Convert JSONL data to standardized JSON format.
Supports three modes:
- offline: New JSONL format (test_gt_withtag.jsonl) with category/Modify fields
- CD (Constructive-Decoding): Original YX format
- advi (Adversarial Injection): New adversarial format
Each input line will produce an individual JSON file.
"""
import json
import re
import os
import argparse

def process_jsonl_file_offline(input_file, output_dir):
    """
    处理 offline 格式的 JSONL 文件 (test_gt_withtag.jsonl)。
    将 <Hallucination> 标签统一转为 <HALLUCINATION>（大写），保持下游兼容。

    Args:
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录路径（一个文件夹）
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                filename = data.get("filename", "")
                category = data.get("category", "")
                modify = data.get("Modify", "Yes")
                is_modified = (modify == "Yes")

                # 图片路径包含 category 子目录
                image_path = f"./test/{category}/{filename}"

                original_caption = data.get("GT_description", "")
                description_tag = data.get("description_tag", "")
                pre_recognition = data.get("Pre-recognition", "")

                # 将 <Hallucination> / </Hallucination> 转为大写 <HALLUCINATION>
                hallucinated_caption_with_tags = re.sub(
                    r'<Hallucination>', '<HALLUCINATION>', description_tag
                )
                hallucinated_caption_with_tags = re.sub(
                    r'</Hallucination>', '</HALLUCINATION>', hallucinated_caption_with_tags
                )

                # 优先使用 JSONL 中自带的 hallucination_labels（含分类信息）
                hallucination_labels = data.get("hallucination_labels", [])
                if not hallucination_labels:
                    # 回退：从带标签的文本中提取（无分类信息）
                    for m in re.finditer(r'<HALLUCINATION>(.*?)</HALLUCINATION>', hallucinated_caption_with_tags):
                        hallucination_labels.append({
                            "text": m.group(1),
                            "labels": []
                        })

                # hallucinated_caption = 去掉标签后的文本
                hallucinated_caption = re.sub(
                    r'<HALLUCINATION>|</HALLUCINATION>', '', hallucinated_caption_with_tags
                )

                processed_data = {
                    "id": filename,
                    "image_path": image_path,
                    "original_caption": original_caption,
                    "hallucinated_caption_with_tags": hallucinated_caption_with_tags,
                    "hallucinated_caption": hallucinated_caption,
                    "hallucination_labels": hallucination_labels,
                    "is_modified": is_modified
                }

                file_stem = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{file_stem}.json")

                with open(output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_data, out_f, ensure_ascii=False, indent=2)

                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行 处理错误: {e}")
                continue

    print(f"[offline模式] 处理完成! 共输出 {count} 个 JSON 文件，目录: {output_dir}")


def process_jsonl_file_CD(input_file, output_dir):
    """
    处理 CD (Constructive-Decoding) 格式的 JSONL 文件。

    Args:
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录路径（一个文件夹）
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                image_path = data.get("image_path", "")
                image_id = os.path.basename(image_path)
                original_caption = data.get("initial_caption", "")
                relative_path = os.path.join("./test", image_id)

                processed_data = {
                    "id": image_id,
                    "image_path": relative_path,
                    "original_caption": original_caption,
                    "is_modified": True  # CD 数据都是有幻觉的
                }

                if "model_reply" in data:
                    model_reply = data["model_reply"]
                    if "hallucinated_caption" in model_reply and "hallucination_labels" in model_reply:
                        hallucinated_caption_with_tags = model_reply["hallucinated_caption"]
                        hallucination_labels = model_reply["hallucination_labels"]

                        hallucinated_caption = re.sub(
                            r'<HALLUCINATION>|</HALLUCINATION>',
                            '',
                            hallucinated_caption_with_tags
                        )

                        processed_data["hallucinated_caption_with_tags"] = hallucinated_caption_with_tags
                        processed_data["hallucinated_caption"] = hallucinated_caption
                        processed_data["hallucination_labels"] = hallucination_labels

                file_stem = os.path.splitext(image_id)[0]
                output_path = os.path.join(output_dir, f"{file_stem}.json")

                with open(output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_data, out_f, ensure_ascii=False, indent=2)

                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行 处理错误: {e}")
                continue

    print(f"[CD模式] 处理完成! 共输出 {count} 个 JSON 文件，目录: {output_dir}")


def process_jsonl_file_advi(input_file, output_dir):
    """
    处理 advi (Adversarial Injection) 格式的 JSONL 文件。

    Args:
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录路径（一个文件夹）
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                # 提取图片路径和ID
                image_path = data.get("image_path", "")
                image_id = os.path.basename(image_path)
                relative_path = os.path.join("./test", image_id)

                # 提取原始描述
                original_caption = data.get("original_description", "")

                # 提取带标签的幻觉描述（直接在顶层）
                hallucinated_caption_with_tags = data.get("description_tag", "")

                # 移除标签得到纯文本幻觉描述
                hallucinated_caption = re.sub(
                    r'<HALLUCINATION>|</HALLUCINATION>',
                    '',
                    hallucinated_caption_with_tags
                )

                # 提取 hallucination_labels
                # 从 model_reply 字符串中解析 JSON，然后从 edits 提取
                hallucination_labels = []
                if "model_reply" in data and isinstance(data["model_reply"], str):
                    try:
                        # 提取 JSON 代码块
                        model_reply_str = data["model_reply"]
                        json_match = re.search(r'```json\s*\n(.*?)\n```', model_reply_str, re.DOTALL)
                        if json_match:
                            model_reply_json = json.loads(json_match.group(1))
                            edits = model_reply_json.get("edits", [])

                            # 从每个 edit 提取幻觉文本和类型
                            for edit in edits:
                                if edit.get("applied", False):
                                    text = edit.get("final_replacement", "")
                                    attr_type = edit.get("attribute_type", "")
                                    if text:
                                        hallucination_labels.append({
                                            "text": text,
                                            "labels": [attr_type] if attr_type else []
                                        })
                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"警告: 第{line_num}行 解析 model_reply 失败: {e}")

                # 构建输出数据结构
                processed_data = {
                    "id": image_id,
                    "image_path": relative_path,
                    "original_caption": original_caption,
                    "hallucinated_caption_with_tags": hallucinated_caption_with_tags,
                    "hallucinated_caption": hallucinated_caption,
                    "hallucination_labels": hallucination_labels,
                    "is_modified": True  # advi 数据都是有幻觉的
                }

                # 构造输出路径
                file_stem = os.path.splitext(image_id)[0]
                output_path = os.path.join(output_dir, f"{file_stem}.json")

                # 写入单个 JSON 文件
                with open(output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_data, out_f, ensure_ascii=False, indent=2)

                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行 处理错误: {e}")
                continue

    print(f"[advi模式] 处理完成! 共输出 {count} 个 JSON 文件，目录: {output_dir}")


def process_jsonl_file_advi_nodetect(input_file, output_dir):
    """
    处理 advi_nodetect 格式的 JSONL 文件。
    与 advi 模式相同，但取 description_tag_nodetect 而非 description_tag。
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                image_path = data.get("image_path", "")
                image_id = os.path.basename(image_path)
                relative_path = os.path.join("./test", image_id)

                original_caption = data.get("original_description", "")

                # 核心区别：取 description_tag_nodetect
                hallucinated_caption_with_tags = data.get("description_tag_nodetect", "")

                hallucinated_caption = re.sub(
                    r'<HALLUCINATION>|</HALLUCINATION>',
                    '',
                    hallucinated_caption_with_tags
                )

                hallucination_labels = []
                if "model_reply" in data and isinstance(data["model_reply"], str):
                    try:
                        model_reply_str = data["model_reply"]
                        json_match = re.search(r'```json\s*\n(.*?)\n```', model_reply_str, re.DOTALL)
                        if json_match:
                            model_reply_json = json.loads(json_match.group(1))
                            edits = model_reply_json.get("edits", [])
                            for edit in edits:
                                if edit.get("applied", False):
                                    text = edit.get("final_replacement", "")
                                    attr_type = edit.get("attribute_type", "")
                                    if text:
                                        hallucination_labels.append({
                                            "text": text,
                                            "labels": [attr_type] if attr_type else []
                                        })
                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"警告: 第{line_num}行 解析 model_reply 失败: {e}")

                processed_data = {
                    "id": image_id,
                    "image_path": relative_path,
                    "original_caption": original_caption,
                    "hallucinated_caption_with_tags": hallucinated_caption_with_tags,
                    "hallucinated_caption": hallucinated_caption,
                    "hallucination_labels": hallucination_labels,
                    "is_modified": True
                }

                file_stem = os.path.splitext(image_id)[0]
                output_path = os.path.join(output_dir, f"{file_stem}.json")

                with open(output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_data, out_f, ensure_ascii=False, indent=2)

                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行 处理错误: {e}")
                continue

    print(f"[advi_nodetect模式] 处理完成! 共输出 {count} 个 JSON 文件，目录: {output_dir}")


def process_jsonl_file_tldr(input_file, output_dir):
    """
    处理 TLDR (testset_inject) 格式的 JSONL 文件。
    字段: filename, category, original_description, hallucinated_description, hallucination_labels

    Args:
        input_file: 输入 JSONL 文件路径
        output_dir: 输出目录路径（一个文件夹）
    """
    os.makedirs(output_dir, exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())

                filename = data.get("filename", "")
                category = data.get("category", "")

                # 图片路径包含 category 子目录
                image_path = f"./test/{category}/{filename}"

                original_caption = data.get("original_description", "")

                # hallucinated_description 已带 <HALLUCINATION> 标签
                hallucinated_caption_with_tags = data.get("hallucinated_description", "")

                # 移除标签得到纯文本
                hallucinated_caption = re.sub(
                    r'<HALLUCINATION>|</HALLUCINATION>',
                    '',
                    hallucinated_caption_with_tags
                )

                # hallucination_labels 直接可用
                hallucination_labels = data.get("hallucination_labels", [])

                processed_data = {
                    "id": filename,
                    "image_path": image_path,
                    "original_caption": original_caption,
                    "hallucinated_caption_with_tags": hallucinated_caption_with_tags,
                    "hallucinated_caption": hallucinated_caption,
                    "hallucination_labels": hallucination_labels,
                    "is_modified": True  # tldr 数据都是有幻觉的
                }

                file_stem = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{file_stem}.json")

                with open(output_path, 'w', encoding='utf-8') as out_f:
                    json.dump(processed_data, out_f, ensure_ascii=False, indent=2)

                count += 1

            except json.JSONDecodeError as e:
                print(f"警告: 第{line_num}行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"警告: 第{line_num}行 处理错误: {e}")
                continue

    print(f"[tldr模式] 处理完成! 共输出 {count} 个 JSON 文件，目录: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL to per-item JSON files.")
    parser.add_argument("--input_jsonls", required=True, nargs='+', help="List of JSONL files to process.")
    parser.add_argument("--output_dir", required=True, help="Directory to store output JSON files.")
    parser.add_argument("--mode", default="CD", choices=["CD", "advi", "advi_nodetect", "offline", "tldr"],
                        help="Processing mode: CD (Constructive-Decoding), advi (adversarial), advi_nodetect (adversarial without detect tags), offline (new JSONL format), or tldr (TLDR testset inject)")
    args = parser.parse_args()

    # 根据模式选择处理函数
    if args.mode == "CD":
        process_func = process_jsonl_file_CD
    elif args.mode == "advi":
        process_func = process_jsonl_file_advi
    elif args.mode == "advi_nodetect":
        process_func = process_jsonl_file_advi_nodetect
    elif args.mode == "offline":
        process_func = process_jsonl_file_offline
    elif args.mode == "tldr":
        process_func = process_jsonl_file_tldr
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    print(f"\n🔧 Mode: {args.mode}")
    print(f"📂 Output directory: {args.output_dir}\n")

    # 清除旧的格式化文件，避免不同 mode 的残留文件干扰
    if os.path.exists(args.output_dir):
        old_files = [f for f in os.listdir(args.output_dir) if f.endswith(".json")]
        if old_files:
            print(f"🧹 Cleaning {len(old_files)} old JSON files from output directory...")
            for f in old_files:
                os.remove(os.path.join(args.output_dir, f))

    # 处理每个输入文件
    for input_file in args.input_jsonls:
        if os.path.exists(input_file):
            print(f"📄 Processing file: {input_file}")
            process_func(input_file, args.output_dir)
        else:
            print(f"⚠️  Warning: File does not exist: {input_file}")