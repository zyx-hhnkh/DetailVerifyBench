import os
import sys
import html as html_module

# ================= 1. 必须最先解决权限问题 (在 import gradio 之前) =================
current_work_dir = os.getcwd()
custom_cache_dir = os.path.join(current_work_dir, "gradio_cache")
if not os.path.exists(custom_cache_dir):
    os.makedirs(custom_cache_dir, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = custom_cache_dir

# ================= 2. 再导入其他库 =================
import argparse
import json
import re
import gradio as gr

# ================= 解决代理问题 =================
keys_to_remove = ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "no_proxy", "NO_PROXY"]
for key in keys_to_remove:
    if key in os.environ:
        del os.environ[key]

# ================= 配置区域 =================
parser = argparse.ArgumentParser(description="Multi-Model Hallucination Caption Visualizer")
parser.add_argument(
    "--image_root_dir", type=str, required=True,
    help="根目录，存放所有图片"
)
parser.add_argument(
    "--exp_dir", type=str, required=True,
    help="实验总目录，包含多个实验子目录 (如 processing_data_zyx_260317)"
)
args = parser.parse_args()

IMAGE_ROOT_DIR = args.image_root_dir
EXP_DIR = args.exp_dir

# ================= 自动发现实验 =================
def discover_experiments(exp_dir):
    """扫描 exp_dir 下所有子目录，查找包含 seed_42 或 seed_0 结果的实验"""
    experiments = {}  # {exp_name: json_file_path}
    for name in sorted(os.listdir(exp_dir)):
        sub = os.path.join(exp_dir, name)
        if not os.path.isdir(sub):
            continue
        # 优先 seed_42，fallback seed_0
        for seed_file in ["tested_model_output_seed_42.json", "tested_model_output_seed_0.json"]:
            path = os.path.join(sub, "inference_results", seed_file)
            if os.path.isfile(path):
                experiments[name] = path
                break
    return experiments

EXPERIMENTS = discover_experiments(EXP_DIR)
EXP_NAMES = list(EXPERIMENTS.keys())
print(f"发现 {len(EXP_NAMES)} 个实验: {EXP_NAMES}")

# ================= 数据缓存 =================
# {exp_name: {image_id: item}}  按 ID 索引
_DATA_CACHE = {}
# {exp_name: [id1, id2, ...]}  保持原始顺序
_IDS_CACHE = {}

def load_experiment(exp_name):
    """加载实验数据，缓存 {id: item} 索引和有序 ID 列表"""
    if exp_name in _DATA_CACHE:
        return _DATA_CACHE[exp_name], _IDS_CACHE[exp_name]
    file_path = EXPERIMENTS.get(exp_name)
    if not file_path or not os.path.exists(file_path):
        _DATA_CACHE[exp_name] = {}
        _IDS_CACHE[exp_name] = []
        return {}, []
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    index = {item.get("id", ""): item for item in data}
    ids = [item.get("id", "") for item in data]
    _DATA_CACHE[exp_name] = index
    _IDS_CACHE[exp_name] = ids
    return index, ids

# ================= 图片路径解析 =================
_IMAGE_INDEX = None

def _build_image_index():
    index = {}
    for root, _, files in os.walk(IMAGE_ROOT_DIR):
        for fname in files:
            index[fname] = os.path.join(root, fname)
    return index

def get_image_path(item):
    global _IMAGE_INDEX
    raw_path = item.get("image_path", "")
    if "./test/" in raw_path:
        rel_path = raw_path.split("./test/", 1)[1]
    else:
        rel_path = item.get("id", "")
    full_path = os.path.join(IMAGE_ROOT_DIR, rel_path)
    if os.path.isfile(full_path):
        return full_path
    if _IMAGE_INDEX is None:
        _IMAGE_INDEX = _build_image_index()
    filename = os.path.basename(rel_path)
    return _IMAGE_INDEX.get(filename, full_path)

# ================= 格式化 =================
def format_html_tags(text):
    if not text:
        return ""
    pattern = re.compile(r'<HALLUCINATION>(.*?)</HALLUCINATION>')
    replacement = r'<span style="background-color: #ffe6e6; color: #d93025; font-weight: bold; padding: 0 4px; border-radius: 4px;">\1</span>'
    return pattern.sub(replacement, text)

def generate_image_choices(image_ids):
    """生成图片下拉框选项: ['0 (ID: xxx)', '1 (ID: yyy)']"""
    return [f"{idx} (ID: {img_id})" for idx, img_id in enumerate(image_ids)]

# ================= 核心显示逻辑 =================
def update_display(exp_name, image_index):
    """根据实验名和图片索引更新所有显示组件"""
    data, image_ids = load_experiment(exp_name)

    if not image_ids or not exp_name:
        return (None, "无数据", "", "", "", "### Reference", "", "", "", image_index,
                gr.update(value=None))

    total = len(image_ids)
    if image_index < 0:
        image_index = 0
    if image_index >= total:
        image_index = total - 1

    img_id = image_ids[image_index]
    item = data.get(img_id)

    if not item:
        return (None, f"实验 {exp_name} 中无 ID: {img_id}", "", "", "", "### 🎯 Reference", "", "",
                exp_name, image_index, gr.update(value=f"{image_index} (ID: {img_id})"))

    img_path = get_image_path(item)
    original_cap = item.get("original_caption", "")

    # 状态
    is_use_original = item.get("is_use_original", False)
    val_passed = item.get("validation_passed", True)
    val_msg = item.get("validation_message", "No message")

    status_content = f"""
    <div style='margin-bottom: 5px; padding: 5px; background-color: #f0f9ff; border-left: 4px solid #3b82f6;'>
        <strong>Config:</strong> is_use_original = <code>{is_use_original}</code>
    </div>
    """
    if not val_passed:
        status_content += f"""
        <div style='margin-top: 5px; padding: 5px; background-color: #fef2f2; border-left: 4px solid #ef4444; color: #b91c1c;'>
            <strong>Validation Failed:</strong> {val_msg}
        </div>
        """

    hallucinated_cap_html = format_html_tags(item.get("hallucinated_caption_with_tags", ""))

    thinking_raw = item.get("thinking_process", item.get("think", ""))
    thinking_escaped = html_module.escape(thinking_raw).replace("\n", "<br>") if thinking_raw else ""
    thinking = f'<div style="max-height: 400px; overflow-y: auto; padding: 10px; background: #f8f8f8; border-radius: 6px; font-size: 14px; line-height: 1.6; white-space: pre-wrap;">{thinking_escaped}</div>'

    if is_use_original:
        ref_label = "### Reference (Type: Original Caption)"
        ref_content_html = format_html_tags(original_cap)
    else:
        ref_label = "### Reference (Type: GT Hallucinated Caption)"
        ref_content_html = format_html_tags(item.get("gt_hallucinated_caption_with_tags", ""))

    progress_str = f"Progress: {image_index + 1} / {total} (ID: {img_id})"
    dropdown_value = f"{image_index} (ID: {img_id})"

    # 构建下拉框的当前值字符串，例如 "0 (ID: xxx)"
    current_dropdown_value = f"{index} (ID: {img_id})"
    
    # 这里的 choices=None 表示不更新选项列表，只更新选中的值
    dropdown_update = gr.update(value=current_dropdown_value)

    return (
        img_path,
        progress_str,
        original_cap,
        hallucinated_cap_html,
        ref_content_html,
        ref_label,
        thinking,
        status_content,
        exp_name,
        image_index,
        gr.update(value=dropdown_value)
    )

# ================= 事件处理 =================
def on_init():
    """初始化：加载第一个实验，第一张图片"""
    if not EXP_NAMES:
        return [None] * 8 + ["", 0, gr.update()]
    exp_name = EXP_NAMES[0]
    _, image_ids = load_experiment(exp_name)
    choices = generate_image_choices(image_ids)
    result = update_display(exp_name, 0)
    return result[:10] + (gr.update(choices=choices, value=choices[0] if choices else None),)

def on_model_change(exp_name, old_exp_name, image_index):
    """切换模型：更新图片下拉框，按 image_id 保持当前图片"""
    # 尝试用旧实验的当前 image_id 在新实验中定位
    old_data, old_ids = load_experiment(old_exp_name) if old_exp_name else ({}, [])
    old_img_id = old_ids[image_index] if 0 <= image_index < len(old_ids) else None

    _, new_ids = load_experiment(exp_name)
    choices = generate_image_choices(new_ids)

    # 在新实验中查找同一 image_id
    new_index = 0
    if old_img_id and old_img_id in new_ids:
        new_index = new_ids.index(old_img_id)
    elif image_index < len(new_ids):
        new_index = image_index

    result = update_display(exp_name, new_index)
    return result[:10] + (gr.update(choices=choices, value=choices[new_index] if choices else None),)

def on_image_select(exp_name, dropdown_value):
    """从下拉框选择图片"""
    if not dropdown_value:
        return [gr.skip()] * 11
    try:
        selected_index = int(dropdown_value.split(" ")[0])
    except:
        selected_index = 0
    return update_display(exp_name, selected_index)

def on_prev(exp_name, image_index):
    return update_display(exp_name, image_index - 1)

def on_next(exp_name, image_index):
    return update_display(exp_name, image_index + 1)

# ================= 界面构建 =================
with gr.Blocks(title="Multi-Model Hallucination Visualizer", theme=gr.themes.Soft()) as demo:
    state_exp = gr.State(EXP_NAMES[0] if EXP_NAMES else "")
    state_index = gr.State(0)

    gr.Markdown("## Multi-Model Hallucination Visualizer")

    with gr.Row():
        with gr.Column(scale=1):
            model_dropdown = gr.Dropdown(
                choices=EXP_NAMES,
                value=EXP_NAMES[0] if EXP_NAMES else None,
                label="1. Select Model / Experiment"
            )

            image_dropdown = gr.Dropdown(
                label="2. Select Image",
                choices=[],
                interactive=True,
                allow_custom_value=False
            )

            with gr.Row():
                btn_prev = gr.Button("Previous")
                btn_next = gr.Button("Next")

            lbl_progress = gr.Label(label="Current ID info")
            img_display = gr.Image(label="Image", type="filepath", height=400)

        with gr.Column(scale=2):
            html_status = gr.HTML(label="Status Info")

            gr.Markdown("### Original Caption")
            html_original = gr.HTML(label="Original Caption")

            gr.Markdown("### Model Output (Hallucinated Caption)")
            html_hallucinated = gr.HTML(label="Hallucinated Caption")

            lbl_ref_title = gr.Markdown("### Reference")
            html_reference = gr.HTML(label="Reference Caption")

            with gr.Accordion("Thinking Process", open=False):
                txt_thinking = gr.HTML()

    # ================= 事件绑定 =================
    common_outputs = [
        img_display, lbl_progress, html_original, html_hallucinated,
        html_reference, lbl_ref_title, txt_thinking, html_status,
        state_exp, state_index, image_dropdown
    ]

    # 模型切换
    model_dropdown.change(
        fn=on_model_change,
        inputs=[model_dropdown, state_exp, state_index],
        outputs=common_outputs
    )

    # 图片选择
    image_dropdown.input(
        fn=on_image_select,
        inputs=[model_dropdown, image_dropdown],
        outputs=common_outputs
    )

    # 上一张/下一张
    btn_prev.click(
        fn=on_prev,
        inputs=[model_dropdown, state_index],
        outputs=common_outputs
    )
    btn_next.click(
        fn=on_next,
        inputs=[model_dropdown, state_index],
        outputs=common_outputs
    )

    # 初始化
    demo.load(
        fn=on_init,
        inputs=[],
        outputs=common_outputs
    )

if __name__ == "__main__":
    if not os.path.exists(IMAGE_ROOT_DIR):
        print(f"WARNING: image dir '{IMAGE_ROOT_DIR}' does not exist.")
    if not EXPERIMENTS:
        print(f"WARNING: no experiments found in '{EXP_DIR}'.")
    demo.launch(share=False, allowed_paths=[IMAGE_ROOT_DIR])
