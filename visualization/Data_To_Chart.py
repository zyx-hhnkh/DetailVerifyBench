import os
import json
import sys

def fmt(value):
    """格式化数值：保留2位小数。如果是 None 或 '-' 则返回 '-'"""
    if value is None or value == '-':
        return "-"
    try:
        return "{:.2f}".format(float(value))
    except (ValueError, TypeError):
        return "-"

def get_metric(data, path_list, metric_key, missing_list):
    """
    安全地从嵌套字典中获取数据。
    如果路径不存在，将错误信息添加到 missing_list，并返回 '-'
    """
    curr = data
    current_path_str = ""
    
    # 遍历路径寻找节点
    for key in path_list:
        current_path_str = f"{current_path_str} -> {key}" if current_path_str else key
        if curr and isinstance(curr, dict) and key in curr:
            curr = curr[key]
        else:
            missing_list.append(f"{current_path_str}")
            return '-'
    
    # 获取最终的数值
    if isinstance(curr, dict):
        if metric_key in curr:
            return curr[metric_key]
        else:
            missing_list.append(f"{current_path_str} -> {metric_key}")
            return '-'
            
    return curr

def process_file(file_path, root_dir):
    try:
        # 计算相对路径
        rel_path = os.path.relpath(file_path, start=root_dir)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # 检查根节点
        if 'average' not in content:
            return {
                "path": rel_path,
                "error": "Missing root key 'average'"
            }
        
        avg_data = content['average']
        missing_keys = [] # 用于收集该文件所有缺失的指标
        
        # 获取模型名称 (使用文件夹名)
        model_name = os.path.basename(os.path.dirname(file_path))
        
        # ==========================================
        # 准备 Table 1 数据: Overall & Category
        # ==========================================
        
        # 1. Overall - Token Level (P, R, F)
        t_p = get_metric(avg_data, ['token_level'], 'precision', missing_keys)
        t_r = get_metric(avg_data, ['token_level'], 'recall', missing_keys)
        t_f = get_metric(avg_data, ['token_level'], 'f1', missing_keys)
        
        # 2. Overall - Sentence Level (P, R, F)
        s_p = get_metric(avg_data, ['sentence_level'], 'precision', missing_keys)
        s_r = get_metric(avg_data, ['sentence_level'], 'recall', missing_keys)
        s_f = get_metric(avg_data, ['sentence_level'], 'f1', missing_keys)
        
        # 3. Category Specific (Recall only)
        cat_map = [
            'Object Number Hallucination',
            'Object Color Hallucination',
            'Object Category Hallucination',
            'Object Shape Hallucination',
            'Object Material Hallucination',
            'Spatial Relation Hallucination',
            'Scene Hallucination',
            'Camera Hallucination',
            'OCR Hallucination',
            'Image Style Hallucination',
            'Character Identification Hallucination',
            'Counterfactual Hallucination',
            'Other Hallucination'
        ]
        
        cat_values = []
        for cat_key in cat_map:
            val = get_metric(avg_data, ['category_level', cat_key], 'recall', missing_keys)
            cat_values.append(val)

        # 组装 Table 1 行
        row1_parts = [model_name]
        row1_parts.extend([t_p, t_r, t_f])
        row1_parts.extend([s_p, s_r, s_f])
        row1_parts.extend(cat_values)
        
        row1_str = " & ".join([fmt(x) if i > 0 else str(x) for i, x in enumerate(row1_parts)]) + " \\\\"

        # ==========================================
        # 准备 Table 2 数据: Segment & Length
        # ==========================================
        
        # 1. Cap. Seg. - 1st Half
        h1_p = get_metric(avg_data, ['split_half', 'first_half'], 'precision', missing_keys)
        h1_r = get_metric(avg_data, ['split_half', 'first_half'], 'recall', missing_keys)
        h1_f = get_metric(avg_data, ['split_half', 'first_half'], 'f1', missing_keys)

        # 2. Cap. Seg. - 2nd Half
        h2_p = get_metric(avg_data, ['split_half', 'second_half'], 'precision', missing_keys)
        h2_r = get_metric(avg_data, ['split_half', 'second_half'], 'recall', missing_keys)
        h2_f = get_metric(avg_data, ['split_half', 'second_half'], 'f1', missing_keys)
        ##################修改分段长度后，需要再手动调整##############################################
        # 3. Cap. Len. - Short (<100)
        l_s_p = get_metric(avg_data, ['length_stratified_level', 'short (<100)'], 'precision', missing_keys)
        l_s_r = get_metric(avg_data, ['length_stratified_level', 'short (<100)'], 'recall', missing_keys)
        l_s_f = get_metric(avg_data, ['length_stratified_level', 'short (<100)'], 'f1', missing_keys)

        # 4. Cap. Len. - Medium (100-150)
        l_m_p = get_metric(avg_data, ['length_stratified_level', 'medium (100-150)'], 'precision', missing_keys)
        l_m_r = get_metric(avg_data, ['length_stratified_level', 'medium (100-150)'], 'recall', missing_keys)
        l_m_f = get_metric(avg_data, ['length_stratified_level', 'medium (100-150)'], 'f1', missing_keys)

        # 5. Cap. Len. - Long (>=150)
        l_l_p = get_metric(avg_data, ['length_stratified_level', 'long (>=150)'], 'precision', missing_keys)
        l_l_r = get_metric(avg_data, ['length_stratified_level', 'long (>=150)'], 'recall', missing_keys)
        l_l_f = get_metric(avg_data, ['length_stratified_level', 'long (>=150)'], 'f1', missing_keys)

        row2_parts = [model_name]
        row2_parts.extend([h1_p, h1_r, h1_f])
        row2_parts.extend([h2_p, h2_r, h2_f])
        row2_parts.extend([l_s_p, l_s_r, l_s_f])
        row2_parts.extend([l_m_p, l_m_r, l_m_f])
        row2_parts.extend([l_l_p, l_l_r, l_l_f])

        row2_str = " & ".join([fmt(x) if i > 0 else str(x) for i, x in enumerate(row2_parts)]) + " \\\\"

        # 如果有缺失的键，生成错误信息
        error_msg = None
        if missing_keys:
            # 去重
            unique_missing = sorted(list(set(missing_keys)))
            error_msg = f"MISSING METRICS in JSON: {', '.join(unique_missing)}"

        return {
            "path": rel_path,
            "row1": row1_str,
            "row2": row2_str,
            "error": error_msg
        }

    except Exception as e:
        # 捕获文件读取或其他未知错误
        return {
            "path": os.path.relpath(file_path, start=root_dir),
            "error": f"FILE PROCESS ERROR: {str(e)}"
        }

def main():
    if len(sys.argv) < 2:
        print("Usage: python Data_To_Chart.py <directory_path>")
        return

    root_dir = sys.argv[1]
    
    table1_data = []
    table2_data = []
    
    # 递归遍历文件夹
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "final_averaged_metrics.json":
                full_path = os.path.join(root, file)
                result = process_file(full_path, root_dir)
                
                if result:
                    table1_data.append(result)
                    table2_data.append(result)

    # 排序（按路径字母顺序）
    table1_data.sort(key=lambda x: x['path'])
    table2_data.sort(key=lambda x: x['path'])

    # === 输出 Table 1 ===
    print("="*80)
    print("TABLE 1: Overall & Category Recall")
    print("="*80)
    for item in table1_data:
        print(f"Path: {item['path']}")
        if item['error']:
            print(f"Error: {item['error']}")
        else:
            print(item['row1'])
        print("-" * 20)

    # === 输出 Table 2 ===
    print("\n" + "="*80)
    print("TABLE 2: Segment & Length Analysis")
    print("="*80)
    for item in table2_data:
        print(f"Path: {item['path']}")
        if item['error']:
            # 注意：如果文件有错，通常两个表的数据都会受影响，这里再次打印错误以防漏看
            print(f"Error: {item['error']}")
        else:
            print(item['row2'])
        print("-" * 20)

if __name__ == "__main__":
    main()