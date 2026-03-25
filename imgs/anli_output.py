import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 字体配置（仅保留负号修复）
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# 你的JSON文件完整路径
JSON_FILE_PATH = r"D:\GitHub\I-GCG\output\20260324-000649\ours\20260324-000649\log\result_32.json"

try:
    # 读取JSON文件
    with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✅ 成功读取JSON文件：{JSON_FILE_PATH}")
    print(f"📊 原始数据条数：{len(data)} 条")

    # 验证必要字段
    required_fields = ["step", "loss", "is_success", "gen_str_ppl", "text_embedding_similarity"]
    for idx, item in enumerate(data[:5]):
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            print(f"⚠️  第{idx}条数据缺失字段：{missing_fields}")

except FileNotFoundError:
    print(f"❌ 错误：找不到文件 {JSON_FILE_PATH}")
    exit(1)
except json.JSONDecodeError:
    print(f"❌ 错误：{JSON_FILE_PATH} 不是有效的JSON文件")
    exit(1)
except Exception as e:
    print(f"❌ 读取文件错误：{str(e)}")
    exit(1)

# 仅提取前200步数据
data_200 = data[:200]
print(f"📊 展示前{len(data_200)}步数据（最多200步）")

# 提取基础指标（保留完整step序列）
steps = np.array([item["step"] for item in data_200])
losses = np.array([item["loss"] for item in data_200])
is_success = np.array([1 if item["is_success"] else 0 for item in data_200])
gen_str_ppl = np.array([item["gen_str_ppl"] for item in data_200])

# --------------------------
# 核心修改：相似度用反余弦函数（arccos）处理
# --------------------------
text_embedding_similarity = np.array([item["text_embedding_similarity"] for item in data_200])
# 反余弦函数输入范围是[-1,1]，先裁剪异常值（避免计算报错）
text_embedding_similarity = np.clip(text_embedding_similarity, -1.0, 1.0)
# 计算反余弦值（结果范围：0 ~ π）
similarity_arccos = np.arccos(text_embedding_similarity)

# --------------------------
# 拆分成功/失败的坐标（保留step连续性）
# --------------------------
# 失败状态的坐标（每个step单独标记，无连线）
fail_mask = is_success == 0
fail_steps = steps[fail_mask]
fail_losses = losses[fail_mask]
fail_similarity = similarity_arccos[fail_mask]
fail_ppl = gen_str_ppl[fail_mask]

# 成功状态的坐标（每个step单独标记，无连线）
success_mask = is_success == 1
success_steps = steps[success_mask]
success_losses = losses[success_mask]
success_similarity = similarity_arccos[success_mask]
success_ppl = gen_str_ppl[success_mask]

# --------------------------
# 绘图：3行1列（Loss → 反余弦相似度 → PPL）
# 核心：仅用散点标记状态，取消连线，保留step连续性
# --------------------------
fig, axes = plt.subplots(3, 1, figsize=(16, 15))
fig.suptitle('Training Metrics (First 200 Steps) - Arccos Similarity', fontsize=18, fontweight='bold', y=0.98)

# 1. Loss（仅散点标记，无连线）
ax1 = axes[0]
# 失败：红色圆形标记
ax1.scatter(fail_steps, fail_losses, color='#FF4444', s=30, alpha=0.8,
            label='Failed (0)', marker='o', edgecolors='black', linewidth=0.5)
# 成功：绿色方形标记
ax1.scatter(success_steps, success_losses, color='#00C851', s=40, alpha=0.9,
            label='Succeeded (1)', marker='s', edgecolors='black', linewidth=0.5)
# 绘制step参考线（增强连续性）
ax1.vlines(steps, ymin=min(losses) * 0.9, ymax=max(losses) * 1.1, colors='gray', alpha=0.1, linewidth=0.5)

ax1.set_title('Loss by Success Status (Discrete Markers)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Loss Value', fontsize=12)
ax1.grid(True, alpha=0.2)
ax1.legend(fontsize=11, loc='upper right')
ax1.xaxis.set_major_locator(MaxNLocator(8))
ax1.tick_params(axis='both', labelsize=10)
# 强制X轴覆盖所有step，保证连续性
ax1.set_xlim(min(steps) - 1, max(steps) + 1)

# 2. 反余弦相似度（仅散点标记，无连线）
ax2 = axes[1]
ax2.scatter(fail_steps, fail_similarity, color='#FF4444', s=30, alpha=0.8,
            label='Failed (0)', marker='o', edgecolors='black', linewidth=0.5)
ax2.scatter(success_steps, success_similarity, color='#00C851', s=40, alpha=0.9,
            label='Succeeded (1)', marker='s', edgecolors='black', linewidth=0.5)
ax2.vlines(steps, ymin=min(similarity_arccos) * 0.9, ymax=max(similarity_arccos) * 1.1, colors='gray', alpha=0.1,
           linewidth=0.5)

ax2.set_title('Arccos of Text Embedding Similarity (Discrete Markers)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xlabel('Step', fontsize=12)
ax2.set_ylabel('Arccos(Similarity) (0 ~ π)', fontsize=12)  # 标注值域范围
ax2.grid(True, alpha=0.2)
ax2.legend(fontsize=11, loc='upper right')
ax2.xaxis.set_major_locator(MaxNLocator(8))
ax2.tick_params(axis='both', labelsize=10)
ax2.set_xlim(min(steps) - 1, max(steps) + 1)

# 3. PPL（最下方，仅散点标记，无连线）
ax3 = axes[2]
ax3.scatter(fail_steps, fail_ppl, color='#FF4444', s=30, alpha=0.8,
            label='Failed (0)', marker='o', edgecolors='black', linewidth=0.5)
ax3.scatter(success_steps, success_ppl, color='#00C851', s=40, alpha=0.9,
            label='Succeeded (1)', marker='s', edgecolors='black', linewidth=0.5)
ax3.vlines(steps, ymin=min(gen_str_ppl) * 0.9, ymax=max(gen_str_ppl) * 1.1, colors='gray', alpha=0.1, linewidth=0.5)

ax3.set_title('Gen Text PPL (Discrete Markers)', fontsize=14, fontweight='bold', pad=15)
ax3.set_xlabel('Step', fontsize=12)
ax3.set_ylabel('PPL Value', fontsize=12)
ax3.grid(True, alpha=0.2)
ax3.legend(fontsize=11, loc='upper right')
ax3.xaxis.set_major_locator(MaxNLocator(8))
ax3.tick_params(axis='both', labelsize=10)
ax3.set_xlim(min(steps) - 1, max(steps) + 1)

# 调整间距
plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.05)

# 保存图片
save_path = "metrics_200steps_arccos_similarity.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"📸 图表已保存为：{save_path}")

# 显示图表
plt.show()

# 打印统计信息
print("\n📈 前200步 - 离散状态统计（反余弦相似度）：")
success_rate = (len(success_steps) / len(steps)) * 100
print(f"前{len(steps)}步整体成功率：{success_rate:.1f}%")
print(f"成功步数：{len(success_steps)} 步 | 失败步数：{len(fail_steps)} 步")

if len(success_steps) > 0:
    print(f"\n✅ 成功状态均值：")
    print(f"   Loss: {np.mean(success_losses):.4f}")
    print(f"   Arccos(Similarity): {np.mean(success_similarity):.4f}")
    print(f"   PPL: {np.mean(success_ppl):.4f}")

if len(fail_steps) > 0:
    print(f"\n❌ 失败状态均值：")
    print(f"   Loss: {np.mean(fail_losses):.4f}")
    print(f"   Arccos(Similarity): {np.mean(fail_similarity):.4f}")
    print(f"   PPL: {np.mean(fail_ppl):.4f}")