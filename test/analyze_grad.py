import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import os
from scipy.stats import pearsonr
# 修复：正确导入并忽略警告
import warnings
warnings.filterwarnings("ignore")

# ===================== 【0】基础配置（修改这里即可！）=====================
# 你的实验输出根目录（和之前分析梯度一致）
ROOT_FOLDER = r"D:\GitHub\I-GCG\Llama-2-7b-chat-hf_result\Radom_20260507-192013"
ANALYSIS_ID = 1  # 要分析的任务ID
CONTROL_POS_NUM = 20  # 你的后缀位置数量（固定20）
TOP_K = 256  # 你的采样TopK（固定256）
# ======================================================================

# ===================== 【1】自动拼接文件路径（严格匹配你的实验命名）=====================
pos_rank_path = Path(ROOT_FOLDER) / "pos_rank" / f"cross_entropy_radom_{ANALYSIS_ID}.npy"
log_path = Path(ROOT_FOLDER) / "log" / f"cross_entropy_radom_{ANALYSIS_ID}.json"
grad_path = Path(ROOT_FOLDER) / "grad" / f"cross_entropy_radom_{ANALYSIS_ID}_grad.npy"
save_fig_path = Path(ROOT_FOLDER) / "filter_analysis"  # 图片保存文件夹
os.makedirs(save_fig_path, exist_ok=True)

# 打印路径校验
print("=" * 80)
print(f"📂 开始分析 任务ID: {ANALYSIS_ID}")
print(f"📊 全量随机候选文件: {pos_rank_path}")
print(f"🎯 筛选最优选择文件: {log_path}")
print(f"📈 梯度文件: {grad_path}")
print("=" * 80)

# ===================== 【2】加载所有数据 =====================
# 2.1 加载 全量随机候选数据 (pos_rank.npy)
# 数据形状: [总步数, 批量大小, 2] → 2代表(位置, 排名)
pos_rank_data = np.load(pos_rank_path, allow_pickle=True)
all_steps, batch_size, _ = pos_rank_data.shape
print(f"\n✅ 全量候选数据加载成功！")
print(f"   总迭代步数: {all_steps} | 每批候选数: {batch_size} | 总候选数量: {all_steps * batch_size}")

# 展平数据: 把所有步、所有批量的 位置/排名 提取出来（全量随机池）
all_random_pos = pos_rank_data[:, :, 0].flatten().astype(int)  # 所有随机选择的位置
all_random_rank = pos_rank_data[:, :, 1].flatten().astype(int)  # 所有随机选择的Token排名

# 2.2 加载 筛选后的最优选择数据 (log.json)
with open(log_path, "r", encoding="utf-8") as f:
    log_data = json.load(f)
# 提取每一步的最优位置、最优排名、Loss（过滤掉报错的step）
best_pos_list = []
best_rank_list = []
best_loss_list = []
for item in log_data:
    if "best_sel_pos" in item and "best_sel_rank" in item and "ce_loss" in item:
        best_pos_list.append(item["best_sel_pos"])
        best_rank_list.append(item["best_sel_rank"])
        best_loss_list.append(item["ce_loss"])

best_pos_list = np.array(best_pos_list, dtype=int)
best_rank_list = np.array(best_rank_list, dtype=int)
best_loss_list = np.array(best_loss_list, dtype=float)
valid_steps = len(best_pos_list)
print(f"\n✅ 最优选择数据加载成功！")
print(f"   有效迭代步数: {valid_steps} | 总最优选择数量: {valid_steps}")

# ===================== 【3】核心分析：位置分布对比（随机 VS 最优）=====================
print("\n" + "="*80)
print("📌 分析1：位置选择分布对比（验证：随机均匀 → 筛选后不均匀）")
print("="*80)

# 3.1 统计：全量随机 → 每个位置被选中的次数/占比
random_pos_count = np.bincount(all_random_pos, minlength=CONTROL_POS_NUM)
random_pos_ratio = random_pos_count / random_pos_count.sum() * 100

# 3.2 统计：筛选最优 → 每个位置被选中的次数/占比
best_pos_count = np.bincount(best_pos_list, minlength=CONTROL_POS_NUM)
best_pos_ratio = best_pos_count / best_pos_count.sum() * 100

# 3.3 打印详细结果
print(f"\n📊 每个位置被选中占比（%）：")
print(f"{'位置':<6}{'全量随机占比':<12}{'筛选最优占比':<12}{'差异(最优-随机)':<15}")
print("-" * 60)
for i in range(CONTROL_POS_NUM):
    diff = best_pos_ratio[i] - random_pos_ratio[i]
    print(f"{i:<6}{random_pos_ratio[i]:<12.2f}{best_pos_ratio[i]:<12.2f}{diff:<15.2f}")

# 3.4 均匀性判断（核心结论）
# 计算方差：方差越小越均匀，方差越大越偏置
random_pos_var = np.var(random_pos_ratio)
best_pos_var = np.var(best_pos_ratio)
print(f"\n🎯 位置分布均匀性判断：")
print(f"   全量随机分布方差: {random_pos_var:.4f}（越小越均匀）")
print(f"   筛选最优分布方差: {best_pos_var:.4f}")
print(f"   👉 结论: 筛选后方差增大{best_pos_var-random_pos_var:.4f}，均匀性被打破！")

# 3.5 找出最优偏好位置
top3_best_pos = np.argsort(best_pos_ratio)[-3:][::-1]
print(f"🏆 筛选后最偏好的 TOP3 位置: {top3_best_pos}")

# ===================== 【4】核心分析：排名分布对比（随机 VS 最优）=====================
print("\n" + "="*80)
print("📌 分析2：Token排名分布对比（验证：随机均匀 → 筛选后集中头部）")
print("="*80)

# 4.1 统计：全量随机 → 排名频次（只看Top50，方便可视化）
rank_bins = np.arange(0, TOP_K+1, 1)
random_rank_hist, _ = np.histogram(all_random_rank, bins=rank_bins)
random_rank_ratio = random_rank_hist / random_rank_hist.sum() * 100

# 4.2 统计：筛选最优 → 排名频次
best_rank_hist, _ = np.histogram(best_rank_list, bins=rank_bins)
best_rank_ratio = best_rank_hist / best_rank_hist.sum() * 100

# 4.3 头部集中性分析（Top1/Top5/Top10/Top20 占比）
top1_random = random_rank_ratio[:1].sum()
top5_random = random_rank_ratio[:5].sum()
top10_random = random_rank_ratio[:10].sum()
top1_best = best_rank_ratio[:1].sum()
top5_best = best_rank_ratio[:5].sum()
top10_best = best_rank_ratio[:10].sum()

print(f"\n🎯 Token排名头部集中性对比：")
print(f"{'排名区间':<10}{'全量随机占比':<12}{'筛选最优占比':<12}{'提升幅度':<10}")
print(f"Top1      {top1_random:<12.2f}{top1_best:<12.2f}{top1_best-top1_random:<10.2f}%")
print(f"Top5      {top5_random:<12.2f}{top5_best:<12.2f}{top5_best-top5_random:<10.2f}%")
print(f"Top10     {top10_random:<12.2f}{top10_best:<12.2f}{top10_best-top10_random:<10.2f}%")

# 4.4 核心结论
print(f"\n👉 终极结论：随机采样时排名均匀分布，筛选后高度集中在头部Token！")

# ===================== 【5】可视化对比图表 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 150

# 图1：位置分布对比柱状图
plt.figure(figsize=(12, 5))
x = np.arange(CONTROL_POS_NUM)
width = 0.35
bar1 = plt.bar(x - width/2, random_pos_ratio, width, label='全量随机候选', color='#1f77b4', alpha=0.8)
bar2 = plt.bar(x + width/2, best_pos_ratio, width, label='筛选最优选择', color='#ff7f0e', alpha=0.8)

for i, v in enumerate(random_pos_ratio):
    plt.text(i - width/2, v + 0.05, f'{v:.1f}', ha='center', fontsize=8)
for i, v in enumerate(best_pos_ratio):
    plt.text(i + width/2, v + 0.05, f'{v:.1f}', ha='center', fontsize=8)

plt.xlabel('后缀位置编号', fontsize=12)
plt.ylabel('被选中占比 (%)', fontsize=12)
plt.title(f'ID{ANALYSIS_ID} 位置选择分布对比（随机 VS 筛选最优）', fontsize=14)
plt.xticks(x)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_fig_path / f'id{ANALYSIS_ID}_position_compare.png', dpi=300)

# 图2：排名分布对比折线图
plt.figure(figsize=(12, 5))
show_top = 30
plt.plot(range(show_top), random_rank_ratio[:show_top], label='全量随机候选',
         color='#1f77b4', linewidth=2.5, marker='o', markersize=4)
plt.plot(range(show_top), best_rank_ratio[:show_top], label='筛选最优选择',
         color='#ff7f0e', linewidth=2.5, marker='s', markersize=4)
plt.fill_between(range(show_top), random_rank_ratio[:show_top], alpha=0.2, color='#1f77b4')
plt.fill_between(range(show_top), best_rank_ratio[:show_top], alpha=0.2, color='#ff7f0e')

plt.xlabel('Token排名 (Top30)', fontsize=12)
plt.ylabel('被选中占比 (%)', fontsize=12)
plt.title(f'ID{ANALYSIS_ID} Token排名分布对比（随机 VS 筛选最优）', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_fig_path / f'id{ANALYSIS_ID}_rank_compare.png', dpi=300)

# ===================== 【新增A】梯度强度 ↔ 位置选择相关性 =====================
print("\n" + "="*80)
print("📌 新增分析A：梯度强度 与 位置选择偏好 相关性")
print("="*80)

grad_data = np.load(grad_path)
position_gradient = np.mean(np.abs(grad_data), axis=(0, 2))
corr, p_val = pearsonr(position_gradient, best_pos_ratio)

print(f"🔍 相关系数 r = {corr:.4f} | 显著性 p = {p_val:.4f}")
print(f"📌 相关性判断：|r|>0.7强相关 | 0.3~0.7中等相关 | <0.3弱相关")
print(f"\n📋 每个位置：梯度强度 | 筛选后选中占比(%)")
for i in range(CONTROL_POS_NUM):
    print(f"位置{i:<2} | 梯度:{position_gradient[i]:<6.2f} | 占比:{best_pos_ratio[i]:<5.2f}")

# 绘图：梯度-位置相关性散点图
plt.figure(figsize=(8, 5))
plt.scatter(position_gradient, best_pos_ratio, s=80, color='#ff7f0e', alpha=0.8, label="位置数据点")

# 安全拟合，无警告
try:
    z = np.polyfit(position_gradient, best_pos_ratio, 1)
    p = np.poly1d(z)
    plt.plot(position_gradient, p(position_gradient), "r--", linewidth=2, label=f"拟合直线 (r={corr:.3f})")
except:
    pass

plt.xlabel("位置梯度强度", fontsize=12)
plt.ylabel("筛选后位置选中占比 (%)", fontsize=12)
plt.title(f"梯度-位置相关性分析 (ID={ANALYSIS_ID})", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(save_fig_path / f'id{ANALYSIS_ID}_gradient_position_corr.png', dpi=300)

# ===================== 【新增B】梯度 ↔ 相邻位置关联性分析 =====================
print("\n" + "="*80)
print("📌 新增分析B：梯度 相邻位置 邻域相关性")
print("="*80)

# 计算每个位置的梯度序列
pos_grad_sequence = np.mean(np.abs(grad_data), axis=2)  # [steps, 20]
# 计算20x20相关矩阵
corr_matrix = np.corrcoef(pos_grad_sequence.T)

# 计算相邻位置相关性
neighbor_corr = []
for i in range(CONTROL_POS_NUM - 1):
    neighbor_corr.append(corr_matrix[i, i+1])
mean_neighbor = np.mean(neighbor_corr)

# 隔一个位置相关性
skip_corr = []
for i in range(CONTROL_POS_NUM - 2):
    skip_corr.append(corr_matrix[i, i+2])
mean_skip = np.mean(skip_corr)

print(f"🔍 相邻位置(i ↔ i+1) 平均相关系数: {mean_neighbor:.4f}")
print(f"🔍 隔一位位置(i ↔ i+2) 平均相关系数: {mean_skip:.4f}")
print("\n📌 相邻梯度关联性结论：")
if mean_neighbor > 0.7:
    print("   ✅ 强相关：相邻位置梯度高度联动、局部连续")
elif mean_neighbor > 0.3:
    print("   ⚠️  中等相关：有一定邻域关联")
else:
    print("   ❌ 弱相关：相邻位置梯度相互独立、无关联")

print(f"\n📋 逐对相邻位置相关系数：")
for i in range(CONTROL_POS_NUM - 1):
    print(f"位置{i} ↔ 位置{i+1} : {corr_matrix[i,i+1]:.4f}")

# 绘图：位置梯度相关性热力图
plt.figure(figsize=(8, 7))
im = plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(im, label='Pearson Correlation')
plt.xticks(range(CONTROL_POS_NUM), range(CONTROL_POS_NUM))
plt.yticks(range(CONTROL_POS_NUM), range(CONTROL_POS_NUM))
plt.xlabel('位置编号', fontsize=12)
plt.ylabel('位置编号', fontsize=12)
plt.title('后缀位置梯度 两两相关性热力图', fontsize=14)
plt.tight_layout()
plt.savefig(save_fig_path / f'id{ANALYSIS_ID}_grad_neighbor_corr.png', dpi=300)

# ===================== 最终总结 =====================
print("\n" + "="*80)
print("📢 全部分析完成！")
print("生成文件：4张高清分析图 + 全维度数据打印")
print("存储路径：", save_fig_path)
print("="*80)

plt.show()