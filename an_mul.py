import json
import pathlib
import argparse
import numpy as np

# ===================== 配置 =====================
parser = argparse.ArgumentParser(description="GCG ")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260416-061952')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--loss_type', type=str, default="cross_entropy", choices=["cross_entropy", "cosine"])
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--str_init', type=str, default="adv_init_suffix")
parser.add_argument('--use_multi_target', type=lambda x: x.lower() == 'true', default=True)
args = parser.parse_args()

# ===================== 【自动分析：单成功 → 多失败】的根本原因 =====================
def analyze_single_success_multi_fail_detailed(method_configs, log_dir, ppl_suffix, args, max_id=200):
    # 1. 查找文件前缀
    file_map = {}
    for cfg in method_configs:
        key = (cfg["mu"], cfg["loss_type"])
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        file_map[key] = file_prefix

    file_single = file_map.get(("", "cross_entropy"))
    file_multi = file_map.get(("multi", "cross_entropy"))

    # 2. 寻找符合条件的ID
    target_id = None
    for run_id in range(1, max_id+1):
        f_s = pathlib.Path(f"{file_single}_{run_id}.json")
        f_m = pathlib.Path(f"{file_multi}_{run_id}.json")
        if not f_s.exists() or not f_m.exists():
            continue

        with open(f_s, 'r', encoding='utf-8') as f:
            data_s = json.load(f)
        with open(f_m, 'r', encoding='utf-8') as f:
            data_m = json.load(f)

        succ_s = data_s[-1]["attack_success"]
        succ_m = data_m[-1]["attack_success"]
        if succ_s and not succ_m:
            target_id = run_id
            break

    if not target_id:
        print("❌ 未找到单成功、多失败的ID")
        return

    print("\n" + "=" * 120)
    print(f"🎯 找到问题 ID = {target_id} | 单目标 ✅ 成功 | 多目标 ❌ 失败")
    print("=" * 120)

    # 加载这组日志
    with open(f"{file_single}_{target_id}.json", encoding='utf-8') as f:
        log_s = json.load(f)
    with open(f"{file_multi}_{target_id}.json", encoding='utf-8') as f:
        log_m = json.load(f)

    # --------------------- 单目标最终状态 ---------------------
    s_final = log_s[-1]
    s_ce = s_final["current_cross_entropy"]
    s_cos = s_final["current_cosine_sim"]
    s_success = s_final["attack_success"]

    # --------------------- 多目标最终状态 ---------------------
    m_final = log_m[-1]
    m_ce = m_final["current_cross_entropy"]
    m_cos = m_final["current_cosine_sim"]
    m_success = m_final["attack_success"]

    print(f"📊 最终结果对比")
    print(f"单目标 | 成功={s_success:>1} | CE={s_ce:>7.3f} | Cos={s_cos:>6.3f}")
    print(f"多目标 | 成功={m_success:>1} | CE={m_ce:>7.3f} | Cos={m_cos:>6.3f}")
    print("-" * 120)

    # --------------------- 多目标切换统计 ---------------------
    switch_times = 0
    stay_main = 0
    ce_jump_steps = []
    last_ce = None

    for step in log_m:
        idx = step.get("current_target_index", "0")
        ce = step["current_cross_entropy"]

        if idx != "0":
            switch_times += 1
        else:
            stay_main += 1

        if last_ce is not None and abs(ce - last_ce) > 0.5:
            ce_jump_steps.append(step["step"])
        last_ce = ce

    total_steps = len(log_m)
    switch_ratio = switch_times / total_steps * 100

    print(f"🔄 多目标切换分析")
    print(f"总步数 {total_steps:>3} | 主目标 {stay_main:>3} | 切换 {switch_times:>3} | 切换比例 {switch_ratio:.1f}%")
    print(f"📉 CE 剧烈跳变步数: {len(ce_jump_steps)} 步")
    if len(ce_jump_steps) > 0:
        print(f"🔍 跳变发生在 steps: {ce_jump_steps[:10]} ...")

    print("-" * 120)

    # --------------------- 自动结论（完全基于日志） ---------------------
    print("\n📌 【自动分析结论：为什么多目标变差】")
    print("=" * 120)

    ce_bad = m_ce > s_ce * 2
    cos_bad = m_cos < s_cos * 0.7
    switch_a_lot = switch_ratio > 10

    reasons = []
    if switch_a_lot:
        reasons.append(f"✅ 目标频繁切换（{switch_ratio:.1f}% 步数在切换）→ 优化方向混乱")
    if len(ce_jump_steps) > 5:
        reasons.append(f"✅ 损失函数剧烈震荡 {len(ce_jump_steps)} 次 → 无法收敛")
    if ce_bad:
        reasons.append(f"✅ 多目标最终 CE={m_ce:.3f}，远高于单目标={s_ce:.3f} → 优化失效")
    if cos_bad:
        reasons.append(f"✅ 多目标 Cos={m_cos:.3f}，远低于单目标={s_cos:.3f} → 匹配失败")

    for r in reasons:
        print(f"  - {r}")

    print("\n🔥 最终结论（100% 来自日志）：")
    if len(reasons) >= 2:
        print("   多目标失败原因：**目标切换 + 损失震荡** → 优化完全发散，攻击失效！")
    elif switch_a_lot:
        print("   多目标失败原因：**目标频繁切换** → 梯度方向冲突，攻击失效！")
    else:
        print("   多目标失败原因：**优化不充分，未收敛到有效后缀**")

    print("=" * 120)
    return target_id
# ===================== 核心诊断函数 =====================
def diagnose_multi_target_issue(method_configs, log_dir, ppl_suffix, args, max_id=200):
    print("\n" + "=" * 120)
    print("🔥  多目标 & Contrast Loss 深度详细诊断报告（逐样本对比）")
    print("=" * 120)

    id_sets = []
    file_map = {}
    for cfg in method_configs:
        key = (cfg["mu"], cfg["loss_type"])
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        file_map[key] = file_prefix

        ids = set()
        for run_id in range(1, max_id + 1):
            f_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            if f_path.exists():
                ids.add(run_id)
        id_sets.append(ids)

    common_ids = sorted(set.intersection(*id_sets))
    if not common_ids:
        print("❌ 没有找到共同样本ID")
        return

    file_single_ce = file_map.get(("", "cross_entropy"))
    file_single_cont = file_map.get(("", "contrast"))
    file_multi_ce = file_map.get(("multi", "cross_entropy"))

    print(f"✅ 共用对比样本数：{len(common_ids)}")
    print("=" * 120)

    total_steps_multi = 0
    total_stayed = 0
    total_switched = 0
    all_stat = []

    for run_id in common_ids:
        f_ce = pathlib.Path(f"{file_single_ce}_{run_id}.json")
        f_ct = pathlib.Path(f"{file_single_cont}_{run_id}.json")
        f_mu = pathlib.Path(f"{file_multi_ce}_{run_id}.json")

        with open(f_ce, 'r', encoding='utf-8') as f:
            d_ce = json.load(f)
        with open(f_ct, 'r', encoding='utf-8') as f:
            d_ct = json.load(f)
        with open(f_mu, 'r', encoding='utf-8') as f:
            d_mu = json.load(f)

        stayed, switched = 0, 0
        for step in d_mu:
            t_str = step.get('current_target_index')
            t = int(t_str) if t_str is not None else 0
            if t == 0:
                stayed += 1
            else:
                switched += 1
        total_stayed += stayed
        total_switched += switched
        total_steps_multi += len(d_mu)

        ce_final = d_ce[-1]
        ce_step = ce_final.get('step', 0)
        ce_success = ce_final.get('attack_success', False)
        ce_ce = ce_final.get('current_cross_entropy', 999)
        ce_cos = ce_final.get('current_cosine_sim', -1)

        ct_final = d_ct[-1]
        ct_step = ct_final.get('step', 0)
        ct_success = ct_final.get('attack_success', False)
        ct_ce = ct_final.get('current_cross_entropy', 999)
        ct_cos = ct_final.get('current_cosine_sim', -1)

        mu_final = d_mu[-1]
        mu_step = mu_final.get('step', 0)
        mu_success = mu_final.get('attack_success', False)
        mu_ce = mu_final.get('current_cross_entropy', 999)
        mu_cos = mu_final.get('current_cosine_sim', -1)

        all_stat.append([
            ce_step, ce_success, ce_ce, ce_cos,
            ct_step, ct_success, ct_ce, ct_cos,
            mu_step, mu_success, mu_ce, mu_cos,
            stayed, switched
        ])

        print(f"🆔 ID {run_id:>3d}")
        print(f"   单CE    | 步:{ce_step:>4d} | 成功:{ce_success!r:<5} | CE:{ce_ce:6.3f} | Cos:{ce_cos:6.3f}")
        print(f"   单Cont  | 步:{ct_step:>4d} | 成功:{ct_success!r:<5} | CE:{ct_ce:6.3f} | Cos:{ct_cos:6.3f}")
        print(f"   多CE    | 步:{mu_step:>4d} | 成功:{mu_success!r:<5} | CE:{mu_ce:6.3f} | Cos:{mu_cos:6.3f}")
        print(f"   多目标切换 | 主目标:{stayed:>4d} | 切换:{switched:>4d} | 总步:{len(d_mu)}")
        print("-" * 120)

    stat = np.array(all_stat, dtype=object)
    N = len(stat)

    ce_mean_step = np.mean([x[0] for x in all_stat])
    ce_rate = np.sum([x[1] for x in all_stat]) / N * 100
    ce_mean_ce = np.mean([x[2] for x in all_stat])
    ce_mean_cos = np.mean([x[3] for x in all_stat])

    ct_mean_step = np.mean([x[4] for x in all_stat])
    ct_rate = np.sum([x[5] for x in all_stat]) / N * 100
    ct_mean_ce = np.mean([x[6] for x in all_stat])
    ct_mean_cos = np.mean([x[7] for x in all_stat])

    mu_mean_step = np.mean([x[8] for x in all_stat])
    mu_rate = np.sum([x[9] for x in all_stat]) / N * 100
    mu_mean_ce = np.mean([x[10] for x in all_stat])
    mu_mean_cos = np.mean([x[11] for x in all_stat])

    stay_rate = total_stayed / total_steps_multi * 100

    print("\n" + "=" * 120)
    print("📊 全局汇总统计")
    print("=" * 120)
    print(f"{'':<10} | {'平均步数':<8} | {'成功率':<8} | {'平均CE':<8} | {'平均Cos':<8}")
    print(f"单CE       | {ce_mean_step:<8.1f} | {ce_rate:<8.1f} | {ce_mean_ce:<8.3f} | {ce_mean_cos:<8.3f}")
    print(f"单Cont     | {ct_mean_step:<8.1f} | {ct_rate:<8.1f} | {ct_mean_ce:<8.3f} | {ct_mean_cos:<8.3f}")
    print(f"多CE       | {mu_mean_step:<8.1f} | {mu_rate:<8.1f} | {mu_mean_ce:<8.3f} | {mu_mean_cos:<8.3f}")
    print("=" * 120)

    # ===================== 动态智能结论（永不写死）=====================
    print("\n💡 动态分析结论（根据实际结果自动判断）")
    print("-" * 120)

    # 1. 损失对比
    ce_improve = ce_mean_ce - ct_mean_ce
    if ce_improve > 0.001:
        print(f"✅ 损失优化：Contrast 损失显著更低，优化值 = {ce_improve:.3f}")
    elif ce_improve < -0.001:
        print(f"❌ 损失恶化：Contrast 损失更高，差值 = {abs(ce_improve):.3f}")
    else:
        print(f"➖ 损失几乎一致：差值 = {ce_improve:.3f}")

    # 2. 成功率对比
    success_diff = ct_rate - ce_rate
    if success_diff > 1:
        print(f"✅ 成功率提升：Contrast 更高，+{success_diff:.1f}%")
    elif success_diff < -1:
        print(f"❌ 成功率下降：Contrast 更低，{success_diff:.1f}%")
    else:
        print(f"➖ 成功率完全相同：{ce_rate:.1f}% ↔ {ct_rate:.1f}%")

    # 3. 收敛速度对比
    step_diff = ct_mean_step - ce_mean_step
    if step_diff < -5:
        print(f"✅ 收敛更快：步数减少 {abs(step_diff):.1f} 步")
    elif step_diff > 5:
        print(f"❌ 收敛更慢：步数增加 {step_diff:.1f} 步")
    else:
        print(f"➖ 收敛速度基本一致：相差 {abs(step_diff):.1f} 步")

    print("-" * 120)
    print("📌 最终综合结论：", end=" ")

    # 综合判断
    has_loss_improve = ce_improve > 0.001
    has_success_up = success_diff > 1
    has_speed_up = step_diff < -5

    if has_success_up or has_speed_up:
        print("✅ Contrast 有效提升攻击效果！")
    elif has_loss_improve and not has_success_up and not has_speed_up:
        print("⚠️ Contrast 仅优化了损失数值，成功率与速度无提升，无实际攻击增益")
    else:
        print("⚠️ Contrast 未带来有效提升，效果与基线一致或更差")

    print("=" * 120)

# ===================== 【精准定位：传入ID，逐Step对比CE，找到第一次分叉】=====================
def analyze_diverge_by_id(target_id, method_configs, log_dir, ppl_suffix, args, ce_thresh=0.1):
    print("\n" + "=" * 140)
    print(f"🎯 直接分析问题 ID = {target_id} | 单目标 ✅ vs 多目标 ❌")
    print("=" * 140)

    # 构建文件路径
    file_map = {}
    for cfg in method_configs:
        key = (cfg["mu"], cfg["loss_type"])
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        file_map[key] = file_prefix

    file_single = file_map.get(("", "cross_entropy"))
    file_multi = file_map.get(("multi", "cross_entropy"))

    f_s = pathlib.Path(f"{file_single}_{target_id}.json")
    f_m = pathlib.Path(f"{file_multi}_{target_id}.json")

    with open(f_s, encoding='utf-8') as f:
        log_s = json.load(f)
    with open(f_m, encoding='utf-8') as f:
        log_m = json.load(f)

    print(f"\n📊 逐Step对比 CE，差异 >= {ce_thresh} 判定为异常")
    print("-" * 140)

    first_diverge_step = None
    first_diverge_detail = None

    min_step = min(len(log_s), len(log_m))
    for i in range(min_step):
        s = log_s[i]
        m = log_m[i]

        step_num = s['step']
        ce_s = s['current_cross_entropy']
        ce_m = m['current_cross_entropy']
        diff = abs(ce_s - ce_m)

        # 多目标是否切换目标
        target_idx = m.get("current_target_index", "0")
        switched = target_idx != "0"

        # 第一次出现差异
        if diff >= ce_thresh and first_diverge_step is None:
            first_diverge_step = step_num
            first_diverge_detail = (ce_s, ce_m, diff, switched)

            print(f"❌ 【第一次出现差异】Step {step_num:>3}")
            print(f"   单CE: {ce_s:.4f}   |   多CE: {ce_m:.4f}   |   差: {diff:.4f}")
            print(f"   多目标是否切换: {switched} (target_idx={target_idx})")
            print("-" * 140)

    # ===================== 最终结论 =====================
    print("\n🔥 精准结论：")
    if first_diverge_step:
        ce_s, ce_m, diff, switched = first_diverge_detail
        print(f"✅ 攻击从 **Step {first_diverge_step}** 开始出现偏差！")
        print(f"✅ 此时单目标CE = {ce_s:.4f}，多目标CE = {ce_m:.4f}，相差 {diff:.4f}")
        if switched:
            print(f"❌ 原因：**这一步切换了目标** → CE直接偏离")
        else:
            print(f"❌ 原因：**没有切换目标，但CE已经偏离** → 优化过程本身出现差异")
    else:
        print(f"✅ 全程CE几乎一致，失败是因为最后没收敛")

    print("=" * 140)
    return first_diverge_step

# ===================== 运行 =====================
if __name__ == "__main__":
    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    method_configs = [
        {"name": "single_ce", "con_loss": "", "mu": "", "loss_type": "cross_entropy"},
        {"name": "single_cont", "con_loss": "", "mu": "", "loss_type": "contrast"},
        {"name": "multi_ce", "con_loss": "", "mu": "multi", "loss_type": "cross_entropy"}
    ]

    diagnose_multi_target_issue(method_configs, log_dir, ppl_suffix, args, max_id=200)

    bad_id = analyze_single_success_multi_fail_detailed(method_configs, log_dir, ppl_suffix, args, max_id=200)

    # 2. 直接用这个ID分析【哪一步出问题】
    if bad_id is not None:
        analyze_diverge_by_id(bad_id, method_configs, log_dir, ppl_suffix, args, ce_thresh=0.1)