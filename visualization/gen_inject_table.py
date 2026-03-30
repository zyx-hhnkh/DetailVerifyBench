import json
import os

BASE_DIR = "processing_data_zyx_260311"
# K=0 -> n1, K=1 -> n2, K=2 -> n3
K_MAP = {0: "n1", 1: "n2", 2: "n3"}
MODELS = {
    "gpt52": "gpt-5.2",
    "qwen3vl8b": "Qwen3-VL-8B",
}

results = {}  # results[k][model_key] = {precision, recall, f1, iou}

for k, n in K_MAP.items():
    results[k] = {}
    for model_key, model_name in MODELS.items():
        exp_name = f"testset_injected_{n}_{model_key}_think_0316"
        metrics_path = os.path.join(BASE_DIR, exp_name, "final_averaged_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                data = json.load(f)
            token = data["average"]["token_level"]
            results[k][model_key] = {
                "precision": token["precision"],
                "recall": token["recall"],
                "f1": token["f1"],
                "iou": token["iou"],
            }
        else:
            results[k][model_key] = None

# Print table
print(f"{'K':<4} {'Token-P(gpt52)':>14} {'Token-P(qwen)':>14} {'Token-R(gpt52)':>14} {'Token-R(qwen)':>14} {'Token-F1(gpt52)':>16} {'Token-F1(qwen)':>14} {'IOU(gpt52)':>11} {'IOU(qwen)':>10}")
print("-" * 120)

for k in [0, 1, 2]:
    row = [str(k)]
    for metric in ["precision", "recall", "f1", "iou"]:
        for model_key in ["gpt52", "qwen3vl8b"]:
            d = results[k].get(model_key)
            if d:
                row.append(f"{d[metric]:.2f}")
            else:
                row.append("-")
    print(f"{row[0]:<4} {row[1]:>14} {row[2]:>14} {row[3]:>14} {row[4]:>14} {row[5]:>16} {row[6]:>14} {row[7]:>11} {row[8]:>10}")

# Also print LaTeX rows
print("\n--- LaTeX rows ---")
for k in [0, 1, 2]:
    parts = [str(k)]
    for metric in ["precision", "recall", "f1", "iou"]:
        for model_key in ["gpt52", "qwen3vl8b"]:
            d = results[k].get(model_key)
            if d:
                parts.append(f"{d[metric]:.2f}")
            else:
                parts.append("-")
    print(" & ".join(parts) + r" \\")
