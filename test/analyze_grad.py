import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

# ===================== 路径 =====================
GRAD_FOLDER = r"D:\GitHub\I-GCG\grad_logs\20260423_182025"
# ================================================

# 加载文件
grad_files = sorted(Path(GRAD_FOLDER).glob("grad_original_*.pt"))
print(f"\n📂 找到梯度文件: {len(grad_files)} 个")

if len(grad_files) == 0:
    print("❌ 没找到文件")
    exit()

# 加载 + 清理
grads = []
for f in grad_files:
    g = torch.load(f, map_location="cpu")
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    grads.append(g)

n_steps = len(grads)
n_pos, vocab_size = grads[0].shape
g_last = grads[-1].numpy()

# ========================= 【1】位置重要性统计 =========================
print("\n==================================================")
print("📊 1. 位置重要性（梯度 L1 范数，越大越重要）")
print("==================================================")
pos_mag = []
for i in range(n_pos):
    mag = np.abs(g_last[i]).sum()
    pos_mag.append(mag)
    print(f"   位置 {i} = {mag:.4f}")

most_important_pos = np.argmax(pos_mag)
least_important_pos = np.argmin(pos_mag)
print(f"👉 最重要位置: 位置 {most_important_pos}")
print(f"👉 最不重要位置: 位置 {least_important_pos}")

# ========================= 【2】梯度分布统计（集中/均匀） =========================
print("\n==================================================")
print("📊 2. 梯度分布（Top10 占比，越高 = 越集中）")
print("==================================================")
for i in range(n_pos):
    vals = np.sort(np.abs(g_last[i]))[::-1]
    top1_ratio = vals[0] / (vals.sum() + 1e-8)
    top5_ratio = vals[:5].sum() / (vals.sum() + 1e-8)
    top10_ratio = vals[:10].sum() / (vals.sum() + 1e-8)
    print(f"   位置 {i} | Top1={top1_ratio:.1%} | Top5={top5_ratio:.1%} | Top10={top10_ratio:.1%}")

print("👉 判断：>50% = 极集中 | 20~50% = 较集中 | <20% = 均匀")

# ========================= 【3】熵统计（随机是否更好） =========================
print("\n==================================================")
print("📊 3. 梯度熵（越高 = 越均匀 = 随机越好）")
print("==================================================")

def entropy(x):
    x = np.abs(x)
    s = x.sum()
    if s < 1e-6:
        return 0.0
    x = x / s
    x = np.clip(x, 1e-5, 1)
    return -np.sum(x * np.log(x))

entropy_list = []
for i in range(n_pos):
    e = entropy(g_last[i])
    entropy_list.append(e)
    print(f"   位置 {i} = {e:.4f}")

mean_ent = np.mean(entropy_list)
print(f"👉 平均熵: {mean_ent:.4f}")
print("👉 判断：>2.0 = 均匀（随机更好） | <1.0 = 集中（贪心更好）")

# ========================= 【4】PCA 方差解释 =========================
print("\n==================================================")
print("📊 4. PCA 降维方差（梯度差异大小）")
print("==================================================")
gn = np.nan_to_num(g_last)
pca = PCA(n_components=2)
pca.fit(gn)
var1, var2 = pca.explained_variance_ratio_
print(f"   主方向1 方差: {var1:.1%}")
print(f"   主方向2 方差: {var2:.1%}")
print(f"   累计解释: {var1+var2:.1%}")
print("👉 >80% = 梯度差异明显 | <50% = 差异很小")

# ========================= 画图 =========================
# 1. 位置重要性
plt.figure(figsize=(10,4))
for i in range(n_pos):
    mags = [np.abs(g[i].numpy()).sum() for g in grads]
    plt.plot(mags, label=f"Pos {i}")
plt.title("Position Importance")
plt.legend()
plt.savefig(f"{GRAD_FOLDER}/01_importance.png")

# 2. 梯度分布
plt.figure(figsize=(10,4))
for i in range(min(n_pos,4)):
    v = np.sort(np.abs(g_last[i]))[::-1][:100]
    plt.plot(v, label=f"Pos {i}")
plt.title("Token Gradient Distribution")
plt.legend()
plt.savefig(f"{GRAD_FOLDER}/02_dist.png")

# 3. 熵
plt.figure(figsize=(10,4))
for i in range(n_pos):
    es = [entropy(g[i].numpy()) for g in grads]
    plt.plot(es, label=f"Pos {i}")
plt.title("Entropy")
plt.legend()
plt.savefig(f"{GRAD_FOLDER}/03_entropy.png")

# 4. PCA
plt.figure(figsize=(6,6))
vec = pca.transform(gn)
for i in range(n_pos):
    plt.scatter(vec[i,0], vec[i,1], s=100, label=f"Pos {i}")
    plt.text(vec[i,0]+0.01, vec[i,1]+0.01, f"{i}")
plt.title("PCA")
plt.savefig(f"{GRAD_FOLDER}/04_pca.png")

print("\n🎉 全部完成！图片已保存 + 数值已打印！")
plt.show()