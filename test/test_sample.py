import torch
import numpy as np
import matplotlib.pyplot as plt
from attack_llm_core_base_select_mothed import sample_control_weighted

# ===================== 随机采样 —— 完全不动 =====================
def sample_control(control_toks, grad, batch_size, topk=128, temp=1, not_allowed_tokens=None):
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = np.inf

    topk_result = (-grad).topk(topk, dim=1)
    top_indices = topk_result.indices

    control_toks = control_toks.to(grad.device)
    original_control_toks = control_toks.repeat(batch_size, 1)

    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size,
        device=grad.device
    ).type(torch.int64)

    rand_idx = torch.randint(0, topk, (batch_size, 1), device=grad.device)
    selected_topk = torch.index_select(top_indices, dim=0, index=new_token_pos)
    new_token_val = torch.gather(input=selected_topk, dim=1, index=rand_idx)

    new_token_pos_u = new_token_pos.unsqueeze(-1)
    new_control_toks = original_control_toks.scatter_(dim=1, index=new_token_pos_u, src=new_token_val)
    unique_count = len(torch.unique(new_control_toks, dim=0))
    print(f"完全随机----生成了 {batch_size} 条，不重复数量：{unique_count}")

    return rand_idx, new_control_toks





# ===================== 测试输入 =====================
def build_test_inputs(seq_len=20, vocab_size=32000, device="cuda"):
    control_toks = torch.randint(0, vocab_size, (seq_len,), device=device)
    grad = torch.randn(seq_len, vocab_size, device=device)
    return control_toks, grad


# ===================== 双图绘制：纵坐标完全一致 =====================
def plot_batch_scatter(rand_ranks, weighted_ranks, common_topk):
    plt.rcParams['font.family'] = ['Arial']
    batch_size = len(rand_ranks)
    x = np.arange(batch_size)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    y_min, y_max = -5, common_topk

    # ---------- 左图：随机采样 ----------
    ax1.scatter(x, rand_ranks, s=6, alpha=0.7, color="#3377bb")
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel("Batch Index")
    ax1.set_ylabel(f"Selected Rank (0 ~ {common_topk})")
    ax1.set_title("Random Sampling", fontweight="bold")
    ax1.grid(alpha=0.3)

    # ---------- 右图：加权采样 ----------
    ax2.scatter(x, weighted_ranks, s=6, alpha=0.9, color="#ff6600")
    ax2.set_ylim(y_min, y_max)  # 与左图完全一样
    ax2.set_xlabel("Batch Index")
    ax2.set_title("Weighted Sampling (Adaptive TopK)", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.suptitle(f"Batch Size = {batch_size} | Common Y-Axis = {common_topk}", fontweight="bold", fontsize=15)
    plt.tight_layout()
    plt.show()

# ===================== 5次实验合并绘图 =====================
def plot_multiple_trials(rand_list, weight_list, common_topk, test_times=5):
    plt.rcParams['font.family'] = ['Arial']
    BATCH = len(rand_list[0])
    x = np.arange(BATCH)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    y_min, y_max = -5, common_topk

    colors = ['#3377bb', '#4488cc', '#5599dd', '#66aaff', '#77bbff']
    # 左图：5次随机采样
    for i in range(test_times):
        ax1.scatter(x, rand_list[i], s=5, alpha=0.6, color=colors[i], label=f'Trial {i+1}')
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('Batch Index (0~%d)' % BATCH)
    ax1.set_ylabel(f'Selected Rank (0 ~ {common_topk})')
    ax1.set_title(f'Random Sampling (5 Trials)', fontweight='bold', fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')

    colors = ['#ff6600', '#ff7711', '#ff8822', '#ff9933', '#ffaa44']
    # 右图：5次加权采样
    for i in range(test_times):
        ax2.scatter(x, weight_list[i], s=5, alpha=0.6, color=colors[i], label=f'Trial {i+1}')
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('Batch Index (0~%d)' % BATCH)
    ax2.set_title(f'Weighted Sampling (Adaptive TopK, 5 Trials)', fontweight='bold', fontsize=14)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')

    plt.suptitle(f'Batch={BATCH} | {test_times} Trials | Common Y-Axis={common_topk}',
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.show()
# ===================== 主函数 =====================
# ===================== 主函数：循环5次 =====================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH = 512
    TOPK = 128
    TEST_TIMES = 5

    rand_list = []       # 保存 5 次随机结果
    weight_list = []     # 保存 5 次加权结果
    adaptive_topk_list = []

    print(f"🚀 开始重复实验 {TEST_TIMES} 次...\n")

    for i in range(TEST_TIMES):
        print(f"▶ 第 {i+1}/{TEST_TIMES} 次实验...")
        control_toks, grad = build_test_inputs(seq_len=20, device=device)

        # 随机
        ranks_rand, _ = sample_control(control_toks, grad, BATCH, TOPK)
        r_rand = ranks_rand.cpu().numpy().squeeze()

        # 加权
        ranks_weight, _, adaptive_topk = sample_control_weighted(control_toks, grad, BATCH, TOPK)
        r_weight = ranks_weight.cpu().numpy().squeeze()

        rand_list.append(r_rand)
        weight_list.append(r_weight)
        adaptive_topk_list.append(adaptive_topk)

    # 统一纵坐标
    common_topk = max(TOPK, max(adaptive_topk_list))

    # 绘图
    plot_multiple_trials(rand_list, weight_list, common_topk, TEST_TIMES)

    print(f"\n✅ 随机TopK: {TOPK}")
    print(f"✅ 5次自适应TopK: {adaptive_topk_list}")
    print(f"✅ 统一纵坐标: 0 ~ {common_topk}")


if __name__ == "__main__":
    main()
