import matplotlib.pyplot as plt
import json

# ===================== JSON 路径 =====================
json_path1 = r"D:\github\I-GCG\output\20260503-115216\log\__cross_entropy__cnn_30.json"
json_path2 = r"D:\github\I-GCG\output\20260503-115216\log\__cross_entropy___30.json"

# ===================== 加载数据 =====================
def load_data(json_path, max_step=100):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:max_step]
    steps = [item["step"] for item in data]
    ce_losses = [item["ce_loss"] for item in data]
    all_batch_mean = [item["all_batch_ce_losses_mean"] for item in data]
    return steps, ce_losses, all_batch_mean

steps1, loss1, batch_mean1 = load_data(json_path1, max_step=100)
steps2, loss2, batch_mean2 = load_data(json_path2, max_step=100)

# ===================== 绘图 =====================
plt.figure(figsize=(10, 5))

# CNN：蓝色系列（实线最优，虚线平均）
plt.plot(steps1, loss1, linewidth=2, color='#1f77b4', label="CNN best")
plt.plot(steps1, batch_mean1, linewidth=1.5, linestyle='--', color='#1f77b4', label="CNN batch mean")

# Random：橙色系列（实线最优，虚线平均）
plt.plot(steps2, loss2, linewidth=2, color='#ff7f0e', label="Random best")
plt.plot(steps2, batch_mean2, linewidth=1.5, linestyle='--', color='#ff7f0e', label="Random batch mean")

plt.title("CE Loss Comparison (First 100 Steps)")
plt.xlabel("Step")
plt.ylabel("CE Loss")
plt.grid(alpha=0.3, linestyle="--")
plt.legend()
plt.tight_layout()

plt.savefig("ce_loss_compare_100steps.png", dpi=300)
plt.show()