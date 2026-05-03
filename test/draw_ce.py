import matplotlib.pyplot as plt
import json

# ===================== JSON 路径 =====================
#RADOM
json_path2 = r"D:\github\I-GCG\output\20260503-145018\log\__cross_entropy___1.json"
#CNN
#json_path1 = r"D:\github\I-GCG\output\20260503-193242\log\__cross_entropy__cnn_1.json"

json_path1 = r"D:\github\I-GCG\output\20260503-211320\log\__cross_entropy__MLP_1.json"

# ===================== 加载数据 =====================
def load_data(json_path, max_step=200):  # <-- 这里改成 200
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data = data[:max_step]
    steps = [item["step"] for item in data]
    ce_losses = [item["ce_loss"] for item in data]
    all_batch_mean = [item["all_batch_ce_losses_mean"] for item in data]
    return steps, ce_losses, all_batch_mean

steps1, loss1, batch_mean1 = load_data(json_path1, max_step=200)  # <-- 200
steps2, loss2, batch_mean2 = load_data(json_path2, max_step=200)  # <-- 200

# ===================== 绘图 =====================
plt.figure(figsize=(12, 6))  # 加宽一点，200步更好看

# CNN：蓝色系列（实线最优，虚线平均）
plt.plot(steps1, loss1, linewidth=2, color='#1f77b4', label="MLP best")
plt.plot(steps1, batch_mean1, linewidth=1.5, linestyle='--', color='#1f77b4', label="MLP batch mean")

# Random：橙色系列（实线最优，虚线平均）
plt.plot(steps2, loss2, linewidth=2, color='#ff7f0e', label="Random best")
plt.plot(steps2, batch_mean2, linewidth=1.5, linestyle='--', color='#ff7f0e', label="Random batch mean")

plt.title("CE Loss Comparison (First 200 Steps)")  # <-- 标题也改成 200
plt.xlabel("Step")
plt.ylabel("CE Loss")
plt.grid(alpha=0.3, linestyle="--")
plt.legend()
plt.tight_layout()

plt.savefig("ce_loss_compare_200steps.png", dpi=300)  # <-- 保存名也改 200
plt.show()