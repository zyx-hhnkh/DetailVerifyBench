#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute precision, recall, F1 and iou metrics.
"""
import json
import re
import nltk
from nltk.tokenize import sent_tokenize
import os
import argparse
import numpy as np
from collections import defaultdict

# 分词方法
# token_pattern = re.compile(
#     r'<HALLUCINATION>|</HALLUCINATION>|'          # ① 标签
#     r"[A-Za-z](?:\.[A-Za-z])+\.?|"                # ② 缩写（U.S.A. / U.K. / Ph.D.）
#     r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|"         # ③ 英文单词 + 连字符/撇号复合词
#     r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+' # ④ 中文汉字、日文平假名、片假名连续字符
# )
token_pattern = re.compile(
    r'<HALLUCINATION>|</HALLUCINATION>|'          # ① 标签
    r"[A-Za-z0-9]+(?:[-'.@_][A-Za-z0-9]+)*|"      # ② 英文复合词
    r"[@#][A-Za-z0-9_]+|"                         # ③ @提及 / #话题
    r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+' # ④ 中日文字符
)

# ================= 增加对于 halluc labels的统计 =================
def mark_positions(text, hallucination_labels=None):
    """
    输入：
      text: 包含 <HALLUCINATION>...</HALLUCINATION> 的字符串
      hallucination_labels: (仅Gold数据有) list, 对应每个tag的详细信息 [{"text":..., "labels":[]}, ...]
    输出：
      pairs: [(token, 0或1), ...]
      positions: [0或1, ...]
      labels: [label_string, ...]  (对应每个token的幻觉类别，无幻觉为 "None")
    """
    # 结构: { "word_content": [ {label_info_1}, {label_info_2} ] }
    label_map = {}
    
    if hallucination_labels:
        for item in hallucination_labels:
            key_text = item.get("text", "").strip().lower()
            key_text = re.sub(r'[^\w\s]', '', key_text) 
            
            if key_text not in label_map:
                label_map[key_text] = []
            label_map[key_text].append(item)

    tag_ranges = []  # [(start, end, label_list), ...]
    tag_matches = list(re.finditer(r"<HALLUCINATION>(.*?)</HALLUCINATION>", text, re.DOTALL))
    
    for m in tag_matches:
        start = m.start() + len("<HALLUCINATION>")
        end = m.end() - len("</HALLUCINATION>")
        raw_content = m.group(1)
        lookup_key = raw_content.strip().lower()
        lookup_key = re.sub(r'[^\w\s]', '', lookup_key)
        
        current_cats = ["Unknown"]
        
        if lookup_key in label_map and len(label_map[lookup_key]) > 0:
            matched_item = label_map[lookup_key].pop(0)
            current_cats = matched_item.get("labels", ["Unknown"])
        
        tag_ranges.append((start, end, current_cats))

    clean_chars = []
    mapping = [] 
    i = 0
    L = len(text)
    while i < L:
        if text.startswith("<HALLUCINATION>", i):
            i += len("<HALLUCINATION>")
        elif text.startswith("</HALLUCINATION>", i):
            i += len("</HALLUCINATION>")
        else:
            clean_chars.append(text[i])
            mapping.append(i)
            i += 1

    clean_text = "".join(clean_chars)

    tokens = []
    for m in token_pattern.finditer(clean_text):
        word = m.group()
        s_clean = m.start()
        # s_clean是 clean_text 中的位置
        s_orig = mapping[s_clean]
        # mapping length is len(clean_text), so mapping[e_clean-1] is safe
        e_orig = mapping[m.end() - 1] + 1
        tokens.append((word, s_orig, e_orig))

    pairs = []
    labels = [] 

    for word, s_o, e_o in tokens:
        label = 1         
        cat_str = "None"  

        for (ts, te, cats) in tag_ranges:
            if not (e_o <= ts or s_o >= te):  
                label = 0
                cat_str = cats[0] if cats else "Unknown"
                break
        
        pairs.append((word, label))
        labels.append(cat_str)

    positions = [p for _, p in pairs]
    return pairs, positions, labels


def load_multiple_json_files_from_dir(dir_path):
    all_items = []
    if not os.path.isdir(dir_path):
        print(f"错误：输入路径不是文件夹: {dir_path}")
        return all_items

    json_files = sorted([f for f in os.listdir(dir_path) if f.endswith(".json")])
    if not json_files:
        print(f"警告：目录中没有找到 .json 文件: {dir_path}")
        return all_items

    print(f"🔍 在 {dir_path} 中找到 {len(json_files)} 个 json 文件")
    for jf in json_files:
        fp = os.path.join(dir_path, jf)
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    all_items.append(data)
                else:
                    print(f"警告：文件 {jf} 不是 dict 格式，跳过")
        except Exception as e:
            print(f"读取文件 {jf} 出错: {e}")

    print(f"📦 合并后共 {len(all_items)} 条数据")
    return all_items


def process_and_save_data(data_list, output_file, is_gold=False, gold_processed=None):
    """
    gold_processed: dict {id: gold_item}，仅在 is_gold=False 时使用。
                    当 test 输出为 "NO HALLUCINATION" 时，生成与 gold 等长的全 1 序列。
    """
    results = []
    no_hal_count = 0
    for item in data_list:
        if "data" in item and "id" in item["data"]:
            item_id = item["data"]["id"]
            text = item["data"]["hallucinated_caption_with_tags"]
            h_labels = item["data"].get("hallucination_labels", None)
        elif "id" in item:
            item_id = item["id"]
            text = item["hallucinated_caption_with_tags"]
            h_labels = item.get("hallucination_labels", None)
        else:
            continue

        # 检测 NO HALLUCINATION 输出（仅 test 数据）
        if not is_gold and gold_processed and text.strip().upper() == "NO HALLUCINATION":
            gold_item = gold_processed.get(item_id)
            if gold_item:
                gold_len = len(gold_item["position_sequence"])
                positions = [1] * gold_len
                pairs = [("NO_HALLUCINATION", 1)] * gold_len
                labels = ["None"] * gold_len
                no_hal_count += 1
            else:
                # fallback: 找不到对应 gold item，使用原有逻辑
                labels_arg = h_labels if is_gold else None
                pairs, positions, labels = mark_positions(text, labels_arg)
        else:
            labels_arg = h_labels if is_gold else None
            pairs, positions, labels = mark_positions(text, labels_arg)

        item_with_markings = item.copy()
        item_with_markings["token_positions"] = pairs
        item_with_markings["position_sequence"] = positions
        item_with_markings["label_sequence"] = labels
        item_with_markings["processed_id"] = item_id
        results.append(item_with_markings)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ 处理并保存中间文件: {output_file} (共 {len(results)} 条)")
    if no_hal_count > 0:
        print(f"   ℹ️  其中 {no_hal_count} 条 NO HALLUCINATION 输出已扩展为与 gold 等长的全 1 序列")
    return results

# ====================================================================

def count_hallucinations(data):
    total_tokens = sum(len(item["position_sequence"]) for item in data)
    hallucinated_tokens = sum(
        sum(1 for x in item["position_sequence"] if x == 0)
        for item in data
    )
    return total_tokens, hallucinated_tokens


def build_domain_index(image_dir):
    """扫描 image_dir 下的子目录，建立 文件名 -> 领域 的映射"""
    domain_map = {}
    if not image_dir or not os.path.isdir(image_dir):
        return domain_map
    for entry in os.listdir(image_dir):
        sub = os.path.join(image_dir, entry)
        if os.path.isdir(sub):
            for fname in os.listdir(sub):
                domain_map[fname] = entry
    return domain_map

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')


def compute_binary_metrics(gold_seq, test_seq):
    if not gold_seq: return 0.0, 0.0, 0.0, 0.0
    if len(gold_seq) != len(test_seq): return 0, 0, 0, 0

    tp = 0 
    fp = 0 
    fn = 0 
    intersection_count = 0
    union_count = 0

    for g, t in zip(gold_seq, test_seq):   # zip将 g和 t一一对应起来
        if g == 0 and t == 0: tp += 1
        elif g == 1 and t == 0: fp += 1
        elif g == 0 and t == 1: fn += 1
        
        if g == 0 and t == 0: intersection_count += 1
        if g == 0 or t == 0: union_count += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    if union_count > 0:
        iou = intersection_count / union_count
    elif union_count == 0 and (tp+fp+fn) == 0:
        iou = 1.0
    else:
        #但是一般不会进入这个逻辑
        iou = 0.0

    return precision, recall, f1, iou


def calculate_metrics(gold_data, test_data, domain_map=None):
    gold_id_map = {item["processed_id"]: item for item in gold_data}
    test_id_map = {item["processed_id"]: item for item in test_data}
    common_ids = set(gold_id_map.keys()) & set(test_id_map.keys())
    if domain_map is None:
        domain_map = {}

    not_match_nums = 0
    # === 新增：定义长度分类阈值 ===
    THRES_SHORT = 100
    THRES_LONG = 150
    
    if not common_ids:
        print("没有找到共同的ID，无法计算指标")
        return None
    print(f"找到 {len(common_ids)} 个共同的ID")

    # metrics_store 使用简写 key: p, r, f1, iou
    # 没写什么level的，都是token level
    metrics_store = {
        "token": {"p": [], "r": [], "f1": [], "iou": []},
        "sentence": {"p": [], "r": [], "f1": []}, 
        "half_1": {"p": [], "r": [], "f1": [], "iou": [], "gt_hal_num":[], "test_hal_num":[], "token_num":[]},
        "half_2": {"p": [], "r": [], "f1": [], "iou": [], "gt_hal_num":[], "test_hal_num":[], "token_num":[]},
        "cats": {},
        "short_cap": {"p": [], "r": [], "f1": [], "iou": []},
        "medium_cap": {"p": [], "r": [], "f1": [], "iou": []},
        "long_cap": {"p": [], "r": [], "f1": [], "iou": []},
        "domain": defaultdict(lambda: {
            "token": {"p": [], "r": [], "f1": [], "iou": []},
            "sentence": {"p": [], "r": [], "f1": []},
        }),
    }
    
    glob_resp_tp = glob_resp_fp = glob_resp_fn = 0

    for item_id in common_ids:
        gold_item = gold_id_map[item_id]
        test_item = test_id_map[item_id]

        g_seq = gold_item["position_sequence"]
        g_cats = gold_item.get("label_sequence", ["None"]*len(g_seq)) # 注意: 使用 label_sequence 对应 mark_positions 的输出
        t_seq = test_item["position_sequence"]
        
        min_len = min(len(g_seq), len(t_seq))
        # ##len不一样的话，这对么？
        # 词换成词组，会导致变长，后面01会错位1-2个。
        if len(g_seq) != len(t_seq):
            not_match_nums+=1
        #     print(f"❗️itemid {item_id} : 01序列长度不同，相差{len(g_seq)}-{len(t_seq)} = {len(g_seq) - len(t_seq)}，已经截断到min_len")
        #     print("="*20)
        #     print(f"golden 01,token pairs:\n { gold_item['token_positions'] }")
        #     print(f"V.S.\ntest 01,token pairs:\n { test_item['token_positions'] }")
        #     print("="*20)
        g_seq = g_seq[:min_len]
        t_seq = t_seq[:min_len]
        g_cats = g_cats[:min_len]

        # 1. Token Level
        p, r, f1, iou = compute_binary_metrics(g_seq, t_seq)
        metrics_store["token"]["p"].append(p)
        metrics_store["token"]["r"].append(r)
        metrics_store["token"]["f1"].append(f1)
        metrics_store["token"]["iou"].append(iou)

        # === 新增：根据 min_len 长度分类存储 ===
        if min_len < THRES_SHORT:
            len_key = "short_cap"
        elif min_len < THRES_LONG:
            len_key = "medium_cap"
        else:
            len_key = "long_cap"
        
        metrics_store[len_key]["p"].append(p)
        metrics_store[len_key]["r"].append(r)
        metrics_store[len_key]["f1"].append(f1)
        metrics_store[len_key]["iou"].append(iou)

        # 2. Sentence Level
        clean_text_for_sent = re.sub(r'<HALLUCINATION>|</HALLUCINATION>', '', gold_item['hallucinated_caption_with_tags'])
        #sentences = sent_tokenize(clean_text_for_sent)  # NLTK 的 sent_tokenize 主要是基于西文标点（如 . ? !）训练的。它无法识别中日文的全角句号（。、！、？）。
        sentences = re.split(r'(?<=[.!?。！？])\s*', clean_text_for_sent)
        sentences = [s for s in sentences if s.strip()] # 过滤掉切分出的空字符串
        
        curr_idx = 0
        sent_tp = sent_fp = sent_fn = 0
        
        for sent in sentences:
            sent_tokens_count = len([tok for tok in token_pattern.findall(sent) 
                                     if tok not in ['<HALLUCINATION>', '</HALLUCINATION>']])
            end_idx = min(curr_idx + sent_tokens_count, min_len)
            
            if curr_idx < end_idx:
                g_sub = g_seq[curr_idx:end_idx]
                t_sub = t_seq[curr_idx:end_idx]
                
                g_has = 1 if any(x==0 for x in g_sub) else 0
                t_has = 1 if any(x==0 for x in t_sub) else 0
                
                if g_has == 1 and t_has == 1: sent_tp += 1
                elif g_has == 0 and t_has == 1: sent_fp += 1
                elif g_has == 1 and t_has == 0: sent_fn += 1
            
            curr_idx = end_idx

        sp = sent_tp / (sent_tp + sent_fp) if (sent_tp + sent_fp) > 0 else 0
        sr = sent_tp / (sent_tp + sent_fn) if (sent_tp + sent_fn) > 0 else 0
        sf1 = 2 * sp * sr / (sp + sr) if (sp + sr) > 0 else 0
        
        metrics_store["sentence"]["p"].append(sp)
        metrics_store["sentence"]["r"].append(sr)
        metrics_store["sentence"]["f1"].append(sf1)

        # Domain Level: 按图片领域分组存储 token 和 sentence 指标
        domain_name = domain_map.get(item_id, "Unknown")
        dm = metrics_store["domain"][domain_name]
        dm["token"]["p"].append(p)
        dm["token"]["r"].append(r)
        dm["token"]["f1"].append(f1)
        dm["token"]["iou"].append(iou)
        dm["sentence"]["p"].append(sp)
        dm["sentence"]["r"].append(sr)
        dm["sentence"]["f1"].append(sf1)

        # 3. Split Half
        mid = min_len // 2
        p1, r1, f1_1, iou1 = compute_binary_metrics(g_seq[:mid], t_seq[:mid])
        metrics_store["half_1"]["p"].append(p1); metrics_store["half_1"]["r"].append(r1)
        metrics_store["half_1"]["f1"].append(f1_1); metrics_store["half_1"]["iou"].append(iou1)
        gt_hal_sum1 = sum(1 for x in g_seq[:mid] if x == 0)
        test_hal_sum1 = sum(1 for x in t_seq[:mid] if x == 0)
        nums1 = len(g_seq[:mid])
        metrics_store["half_1"]["gt_hal_num"].append(gt_hal_sum1)
        metrics_store["half_1"]["test_hal_num"].append(test_hal_sum1)
        metrics_store["half_1"]["token_num"].append(nums1)
        
        p2, r2, f1_2, iou2 = compute_binary_metrics(g_seq[mid:], t_seq[mid:])
        metrics_store["half_2"]["p"].append(p2); metrics_store["half_2"]["r"].append(r2)
        metrics_store["half_2"]["f1"].append(f1_2); metrics_store["half_2"]["iou"].append(iou2)
        gt_hal_sum2 = sum(1 for x in g_seq[mid:] if x == 0)
        test_hal_sum2 = sum(1 for x in t_seq[mid:] if x == 0)
        nums2 = len(g_seq[mid:])
        metrics_store["half_2"]["gt_hal_num"].append(gt_hal_sum2)
        metrics_store["half_2"]["test_hal_num"].append(test_hal_sum2)
        metrics_store["half_2"]["token_num"].append(nums2)

        # 4. Category Level (NumPy Vectorized)
        # 先把列表转为 numpy 数组 (只转一次)
        arr_g_cats = np.array(g_cats[:min_len])
        arr_t_seq = np.array(t_seq[:min_len])

        # 找出所有涉及的类别（排除 None）
        target_cats = set(np.unique(arr_g_cats)) - {"None"}
        
        for cat in target_cats:
            if cat not in metrics_store["cats"]:
                metrics_store["cats"][cat] = {"hits_num": [], "gt_num": []}

            # === NumPy 核心逻辑 ===
            # 1. 生成掩码：找出所有属于当前 cat 的位置 (True/False 向量)
            cat_mask = (arr_g_cats == cat)
            
            # 2. 分母：GT 中该类别的总数 (True 的个数)
            total_gt = np.sum(cat_mask)

            # 3. 分子：在这些位置上，t_seq 为 0 (预测为幻觉) 的数量
            # arr_t_seq[cat_mask] 提取出对应位置的预测值
            caught_hits = np.sum(arr_t_seq[cat_mask] == 0)
            
            metrics_store["cats"][cat]["hits_num"].append(int(caught_hits))
            metrics_store["cats"][cat]["gt_num"].append(int(total_gt))

        # 5. Response Level
        gh = 1 if any(x==0 for x in g_seq) else 0
        th = 1 if any(x==0 for x in t_seq) else 0
        if gh==1 and th==1: glob_resp_tp+=1
        elif gh==0 and th==1: glob_resp_fp+=1
        elif gh==1 and th==0: glob_resp_fn+=1
    
    rp = glob_resp_tp/(glob_resp_tp+glob_resp_fp) if (glob_resp_tp+glob_resp_fp)>0 else 0
    rr = glob_resp_tp/(glob_resp_tp+glob_resp_fn) if (glob_resp_tp+glob_resp_fn)>0 else 0
    rf1 = 2*rp*rr/(rp+rr) if (rp+rr)>0 else 0

    def get_avg(lst): return sum(lst)/len(lst) if lst else 0.0

    gold_total_tokens, gold_hallu_tokens = count_hallucinations([gold_id_map[item_id] for item_id in common_ids])
    test_total_tokens, test_hallu_tokens = count_hallucinations([test_id_map[item_id] for item_id in common_ids])


    result = {
        "hallucination_counts": {
            "gold_total_tokens": gold_total_tokens, "gold_hallu_tokens": gold_hallu_tokens,
            "test_total_tokens": test_total_tokens, "test_hallu_tokens": test_hallu_tokens,
            "avg_gt_total_tokens": gold_total_tokens/len(common_ids), "avg_gt_hal_tokens": gold_hallu_tokens/len(common_ids),
            "avg_test_total_tokens": test_total_tokens/len(common_ids), "avg_test_hal_tokens": test_hallu_tokens/len(common_ids),
            
        },
        "token_level": {
            "precision": get_avg(metrics_store["token"]["p"]),
            "recall":    get_avg(metrics_store["token"]["r"]),
            "f1":        get_avg(metrics_store["token"]["f1"]),
            "iou":       get_avg(metrics_store["token"]["iou"])
        },
        "sentence_level": {
            "precision": get_avg(metrics_store["sentence"]["p"]),
            "recall":    get_avg(metrics_store["sentence"]["r"]),
            "f1":        get_avg(metrics_store["sentence"]["f1"])
        }, 
        "response_level":  {"precision": rp, "recall": rr, "f1": rf1},
        "split_half": {
            "first_half": {
                "precision": get_avg(metrics_store["half_1"]["p"]),
                "recall":    get_avg(metrics_store["half_1"]["r"]),
                "f1":        get_avg(metrics_store["half_1"]["f1"]),
                "iou":       get_avg(metrics_store["half_1"]["iou"]),
                "count":     len(common_ids),
                "avg_gt_hal_count":     sum(metrics_store["half_1"]["gt_hal_num"]) / len(common_ids) if len(common_ids)>0 else 0.0,
                "avg_test_hal_count":     sum(metrics_store["half_1"]["test_hal_num"]) / len(common_ids) if len(common_ids)>0 else 0.0,
            },
            "second_half": {
                "precision": get_avg(metrics_store["half_2"]["p"]),
                "recall":    get_avg(metrics_store["half_2"]["r"]),
                "f1":        get_avg(metrics_store["half_2"]["f1"]),
                "iou":       get_avg(metrics_store["half_2"]["iou"]),
                "count":     len(common_ids),
                "avg_gt_hal_count":     sum(metrics_store["half_2"]["gt_hal_num"]) / len(common_ids) if len(common_ids)>0 else 0.0,
                "avg_test_hal_count":     sum(metrics_store["half_2"]["test_hal_num"]) / len(common_ids) if len(common_ids)>0 else 0.0,
            },
        },
        # === 新增：按长度分层的汇总结果 ===
        "length_stratified_level": {
            "short (<{})".format(THRES_SHORT): {
                "precision": get_avg(metrics_store["short_cap"]["p"]),
                "recall":    get_avg(metrics_store["short_cap"]["r"]),
                "f1":        get_avg(metrics_store["short_cap"]["f1"]),
                "iou":       get_avg(metrics_store["short_cap"]["iou"]),
                "count":     len(metrics_store["short_cap"]["p"])
            },
            "medium ({}-{})".format(THRES_SHORT, THRES_LONG): {
                "precision": get_avg(metrics_store["medium_cap"]["p"]),
                "recall":    get_avg(metrics_store["medium_cap"]["r"]),
                "f1":        get_avg(metrics_store["medium_cap"]["f1"]),
                "iou":       get_avg(metrics_store["medium_cap"]["iou"]),
                "count":     len(metrics_store["medium_cap"]["p"])
            },
            "long (>={})".format(THRES_LONG): {
                "precision": get_avg(metrics_store["long_cap"]["p"]),
                "recall":    get_avg(metrics_store["long_cap"]["r"]),
                "f1":        get_avg(metrics_store["long_cap"]["f1"]),
                "iou":       get_avg(metrics_store["long_cap"]["iou"]),
                "count":     len(metrics_store["long_cap"]["p"])
            }
        },
        "category_level": {},
        "domain_level": {},
        "common_ids_count": len(common_ids)
    }

    for cat, vals in metrics_store["cats"].items():
        hits = sum(vals["hits_num"])
        gt_num = sum(vals["gt_num"])
        result["category_level"][cat] = {
            "recall": hits / gt_num if gt_num > 0 else 0.0,
            "hits_count": int(hits),
            "gt_count": int(gt_num),
        }

    # Domain Level 汇总
    for domain_name, dm in metrics_store["domain"].items():
        result["domain_level"][domain_name] = {
            "token_level": {
                "precision": get_avg(dm["token"]["p"]),
                "recall":    get_avg(dm["token"]["r"]),
                "f1":        get_avg(dm["token"]["f1"]),
                "iou":       get_avg(dm["token"]["iou"]),
            },
            "sentence_level": {
                "precision": get_avg(dm["sentence"]["p"]),
                "recall":    get_avg(dm["sentence"]["r"]),
                "f1":        get_avg(dm["sentence"]["f1"]),
            },
            "common_ids_count": len(dm["token"]["p"]),
        }

    return result

'''
def calculate_average_metrics(metrics_list):
    """
    计算多次运行结果的平均值。
    """
    if not metrics_list: return None
    
    num_seeds = len(metrics_list)
    print(f"📊 Aggregating metrics over {num_seeds} seeds...")

    # ==========================================
    # 1. 定义累加器结构 (Accumulator - 使用全称 key)
    # ==========================================
    agg = {
        "token_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0},
        "sentence_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "split_half": {
            "first_half": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0},
            "second_half": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0},
        },
        "response_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0},
        "hallucination_counts": {
            "gold_total_tokens": 0.0, "gold_hallu_tokens": 0.0,
            "test_total_tokens": 0.0, "test_hallu_tokens": 0.0
        },
        "category_level": {} 
    }

    # ==========================================
    # 2. 遍历每个 Seed 的结果进行累加
    # ==========================================
    for m in metrics_list:
        
        # --- Token Level ---
        agg["token_level"]["precision"] += m["token_level"]["precision"]
        agg["token_level"]["recall"] += m["token_level"]["recall"]
        agg["token_level"]["f1"] += m["token_level"]["f1"]
        agg["token_level"]["iou"] += m["token_level"]["iou"]

        # --- Sentence Level ---
        agg["sentence_level"]["precision"] += m["sentence_level"]["precision"]
        agg["sentence_level"]["recall"] += m["sentence_level"]["recall"]
        agg["sentence_level"]["f1"] += m["sentence_level"]["f1"]

        # --- Split Half ---
        for half in ["first_half", "second_half"]:
            agg["split_half"][half]["precision"] += m["split_half"][half]["precision"]
            agg["split_half"][half]["recall"] += m["split_half"][half]["recall"]
            agg["split_half"][half]["f1"] += m["split_half"][half]["f1"]
            agg["split_half"][half]["iou"] += m["split_half"][half]["iou"]

        # --- Response Level ---
        agg["response_level"]["precision"] += m["response_level"]["precision"]
        agg["response_level"]["recall"] += m["response_level"]["recall"]
        agg["response_level"]["f1"] += m["response_level"]["f1"]

        # --- Hallucination Counts ---
        curr_counts = m.get("hallucination_counts", {})
        for k in agg["hallucination_counts"]:
            agg["hallucination_counts"][k] += curr_counts.get(k, 0)

        # --- Category Level ---
        for cat, vals in m["category_level"].items():
            if cat not in agg["category_level"]:
                agg["category_level"][cat] = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0, "count": 0}
            
            agg["category_level"][cat]["precision"] += vals["precision"]
            agg["category_level"][cat]["recall"] += vals["recall"]
            agg["category_level"][cat]["f1"] += vals["f1"]
            agg["category_level"][cat]["iou"] += vals["iou"]
            agg["category_level"][cat]["count"] += 1 

    # ==========================================
    # 3. 计算平均值 (Division)
    # ==========================================
    final_avg = {
        "token_level": {}, "sentence_level": {}, 
        "split_half": {"first_half": {}, "second_half": {}}, 
        "response_level": {}, "category_level": {},
        "hallucination_counts": {}
    }

    def div_write(target_dict, key, val):
        target_dict[key] = val / num_seeds


    # --- Token ---
    div_write(final_avg["token_level"], "precision", agg["token_level"]["precision"])
    div_write(final_avg["token_level"], "recall", agg["token_level"]["recall"])
    div_write(final_avg["token_level"], "f1", agg["token_level"]["f1"])
    div_write(final_avg["token_level"], "iou", agg["token_level"]["iou"])

    # --- Sentence ---
    div_write(final_avg["sentence_level"], "precision", agg["sentence_level"]["precision"])
    div_write(final_avg["sentence_level"], "recall", agg["sentence_level"]["recall"])
    div_write(final_avg["sentence_level"], "f1", agg["sentence_level"]["f1"])

    # --- Split Half ---
    for half in ["first_half", "second_half"]:
        div_write(final_avg["split_half"][half], "precision", agg["split_half"][half]["precision"])
        div_write(final_avg["split_half"][half], "recall", agg["split_half"][half]["recall"])
        div_write(final_avg["split_half"][half], "f1", agg["split_half"][half]["f1"])
        div_write(final_avg["split_half"][half], "iou", agg["split_half"][half]["iou"])

    # --- Response ---
    div_write(final_avg["response_level"], "precision", agg["response_level"]["precision"])
    div_write(final_avg["response_level"], "recall", agg["response_level"]["recall"])
    div_write(final_avg["response_level"], "f1", agg["response_level"]["f1"])

    # --- Counts ---
    for k in agg["hallucination_counts"]:
        final_avg["hallucination_counts"][k] = agg["hallucination_counts"][k] / num_seeds

    # --- Category ---
    for cat, vals in agg["category_level"].items():
        final_avg["category_level"][cat] = {
            "precision": vals["precision"] / num_seeds,
            "recall": vals["recall"] / num_seeds,
            "f1": vals["f1"] / num_seeds,
            "iou": vals["iou"] / num_seeds
        }

    return final_avg
'''

# 优雅递归处理，加新的指标不需要重写
def calculate_average_metrics(metrics_list):
    """
    递归计算任意嵌套字典结构的平均值。
    能够自动处理动态 Key (如 category_level) 和新增的指标。
    """
    if not metrics_list: return None
    
    num_seeds = len(metrics_list)
    print(f"📊 Aggregating metrics over {num_seeds} seeds...")

    def recursive_avg(dicts_list):
        # 1. 获取当前层级所有出现过的 key (并集)，应对某些 seed 缺失 key 的情况
        all_keys = set().union(*[d.keys() for d in dicts_list])
        avg_dict = {}

        for k in all_keys:
            # 收集当前 key 在所有 seed 中的值
            values = [d[k] for d in dicts_list if k in d]
            
            if not values: continue

            # 取第一个非空值判断类型
            first_val = values[0]

            if isinstance(first_val, dict):
                # 如果是字典，递归处理下一层
                avg_dict[k] = recursive_avg(values)
            elif isinstance(first_val, (int, float)):
                # 如果是数值，直接求和并除以总 seed 数
                # 注意：这里除以 num_seeds 是为了保持和你原逻辑一致（分母为总运行次数）
                avg_dict[k] = sum(values) / num_seeds
            else:
                # 如果是字符串或其他元数据，保留第一个遇到的值 (可选)
                avg_dict[k] = first_val

        return avg_dict

    return recursive_avg(metrics_list)

def main(args):
    # 1. 准备 Gold 数据
    print(f"\n🚀 Step 1: 加载 Gold 数据 from {args.input_gold_dir}")
    raw_gold_data = load_multiple_json_files_from_dir(args.input_gold_dir)
    if not raw_gold_data:
        print("❌ 未加载到 Gold 数据，退出。")
        return

    # 构建 文件名 -> 领域 映射
    domain_map = build_domain_index(getattr(args, 'image_dir', None))
    if domain_map:
        domains = set(domain_map.values())
        print(f"🗂️  已构建领域索引: {len(domain_map)} 个文件, {len(domains)} 个领域 ({', '.join(sorted(domains))})")

    # 处理并保存 Gold 中间结果
    print("\n🛠️  Processing Gold Data to 0/1 sequences...")
    processed_gold = process_and_save_data(raw_gold_data, args.output_gold_processed, is_gold=True)

    all_seed_metrics = []

    # 2. 循环处理每个 Test 文件
    print(f"\n🚀 Step 2: 开始处理 {len(args.input_test_files)} 个测试文件...")
    gold_map = {item["processed_id"]: item for item in processed_gold}

    for idx, test_file_path in enumerate(args.input_test_files):
        print(f"\n--- Processing Seed #{idx+1}: {os.path.basename(test_file_path)} ---")

        with open(test_file_path, 'r', encoding='utf-8') as f:
            raw_test_data = json.load(f)
            if isinstance(raw_test_data, dict): raw_test_data = [raw_test_data]

        base, ext = os.path.splitext(test_file_path)
        processed_test_path = f"{base}_alter_to_01{ext}"

        print(f"🛠️  生成 0/1 序列 -> {processed_test_path}")
        processed_test = process_and_save_data(raw_test_data, processed_test_path, is_gold=False, gold_processed=gold_map)

        metrics = calculate_metrics(processed_gold, processed_test, domain_map=domain_map)
        if metrics:
            all_seed_metrics.append(metrics)
            print(f"✅ Seed #{idx+1} Token-F1: {metrics['token_level']['f1']:.4f}")

    # 3. 计算平均值并输出
    if all_seed_metrics:
        print("\n📊 Calculating Average Metrics across seeds...")
        avg_results = calculate_average_metrics(all_seed_metrics)
        
        final_output = {
            "average": avg_results,
            "details": all_seed_metrics
        }

        print("\n=== 最终平均结果 ===")
        print(json.dumps(avg_results, indent=2, ensure_ascii=False))

        with open(args.output_metrics, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)

        print(f"\n📄 最终平均指标已保存到：{args.output_metrics}")
    else:
        print("❌ 没有有效的结果用于计算平均值")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data and compute averaged metrics across seeds.")
    
    parser.add_argument("--input_gold_dir", required=True, help="Golden 文件夹")
    parser.add_argument("--input_test_files", required=True, nargs='+', help="多个 Test 原始文件路径 (空格分隔)")
    parser.add_argument("--output_gold_processed", required=True, help="中间文件：Golden 处理后的 0/1 json")
    parser.add_argument("--output_metrics", required=True, help="最终结果：包含平均值和详情的 json")
    parser.add_argument("--image_dir", required=False, default=None, help="图片根目录（含领域子目录），用于按领域统计指标")
    
    args = parser.parse_args()

    main(args)