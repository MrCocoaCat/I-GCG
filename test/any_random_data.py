import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

def auto_analyze_by_date_and_id(run_date_dir, data_id, loss_type="cross_entropy", sample_method="radom"):
    """
    独立分析函数：仅需输入实验文件夹 + ID，自动分析 pos_rank 和梯度矩阵
    输出：统计表格 + 分布图 + 2张专业热力图
    """
    # 自动拼接路径（完全匹配你的保存格式）
    base_out = run_date_dir
    pos_rank_path = os.path.join(base_out, "pos_rank", f"{loss_type}_{sample_method}_{data_id}.npy")
    grad_path = os.path.join(base_out, "grad", f"{loss_type}_{sample_method}_{data_id}_grad.npy")
    save_dir = os.path.join(base_out, f"analysis_id_{data_id}")
    os.makedirs(save_dir, exist_ok=True)

    # 加载数据
    print(f"加载文件：{pos_rank_path}")
    print(f"加载文件：{grad_path}")
    pos_rank_data = np.load(pos_rank_path, allow_pickle=True)
    grad_data = np.load(grad_path, allow_pickle=True)

    # 整理 position 和 rank 数据
    all_pos = []
    all_rk = []
    for step in pos_rank_data:
        for p, r in step:
            all_pos.append(p)
            all_rk.append(r)

    # 梯度数据处理：Position × Step 热力图
    grad_mean_per_pos = [g.mean(axis=1) for g in grad_data]
    grad_heat_mat = np.array(grad_mean_per_pos).T

    # ========== 基础统计 ==========
    pos_cnt = Counter(all_pos)
    rk_cnt = Counter(all_rk)
    total_steps = len(pos_rank_data)

    # 保存统计文件
    stats_dict = {
        "run_dir": run_date_dir,
        "id": data_id,
        "total_steps": total_steps,
        "total_sample_num": len(all_pos),
        "most_common_pos": pos_cnt.most_common(1)[0],
        "most_common_rank": rk_cnt.most_common(1)[0]
    }
    pd.DataFrame([stats_dict]).to_csv(os.path.join(save_dir, "basic_stats.csv"), index=False)
    pd.DataFrame(pos_cnt.items(), columns=["position", "count"]).to_csv(os.path.join(save_dir, "pos_count.csv"), index=False)
    pd.DataFrame(rk_cnt.items(), columns=["rank", "count"]).to_csv(os.path.join(save_dir, "rank_count.csv"), index=False)

    # ========== 基础分布图 ==========
    plt.figure(figsize=(12, 7))
    plt.subplot(2, 2, 1)
    plt.bar(pos_cnt.keys(), pos_cnt.values())
    plt.title("Position Select Count")
    plt.xlabel("pos")
    plt.ylabel("count")

    plt.subplot(2, 2, 2)
    plt.bar(rk_cnt.keys(), rk_cnt.values())
    plt.title("Rank Select Count")
    plt.xlabel("rank")

    plt.subplot(2, 2, 3)
    grad_mean_line = [g.mean() for g in grad_data]
    plt.plot(grad_mean_line)
    plt.title("Global Grad Mean Over Steps")
    plt.xlabel("step")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "basic_plot.png"), dpi=200)
    plt.close()

    # ========== 热力图1：Position × Rank 选择频次热力图 ==========
    max_p = max(all_pos) + 1
    max_r = max(all_rk) + 1
    pr_hm = np.zeros((max_p, max_r), dtype=int)
    for p, r in zip(all_pos, all_rk):
        pr_hm[p, r] += 1

    plt.figure(figsize=(14, 6))
    sns.heatmap(pr_hm, cmap="YlGnBu", cbar_kws={"label": "select frequency"})
    plt.title("Position × Rank Heatmap", fontsize=14)
    plt.xlabel("Top-k Rank")
    plt.ylabel("Token Position")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap_pos_rank.png"), dpi=200)
    plt.close()

    # ========== 热力图2：Gradient Position × Step 热力图 ==========
    plt.figure(figsize=(14, 6))
    sns.heatmap(grad_heat_mat, cmap="coolwarm", cbar_kws={"label": "grad mean"})
    plt.title("Gradient Heatmap (Position × Step)", fontsize=14)
    plt.xlabel("Step")
    plt.ylabel("Token Position")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "heatmap_grad_pos_step.png"), dpi=200)
    plt.close()

    print(f"\n✅ 分析完成！结果保存在：{save_dir}")


if __name__ == '__main__':
    """
    主函数入口：只需要修改下面 2 个参数
    """
    # ===================== 请在这里修改参数 =====================
    # 你的实验根目录（日期文件夹）
    RUN_FOLDER = "../Llama-2-7b-chat-hf_result/Radom_20260507-184118"
    # 要分析的样本 ID
    TARGET_ID = 1

    # 执行分析
    auto_analyze_by_date_and_id(
        run_date_dir=RUN_FOLDER,
        data_id=TARGET_ID
    )