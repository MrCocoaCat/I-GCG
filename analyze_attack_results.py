import json
import pathlib
import argparse
import numpy as np

# ===================== 1. 配置解析 =====================
parser = argparse.ArgumentParser(description="GCG 攻击结果自动评估脚本")
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--loss_type', type=str, default="cross_entropy", choices=["cross_entropy", "cosine"])
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--use_multi_target', type=lambda x: x.lower() == 'true', default=True)

args = parser.parse_args()



# ===================== ✅ 正确统计：真正的【切换动作次数】 =====================
def count_real_switch_actions(file_multi, failed_ids):
    print("\n" + "=" * 140)
    print("🔥 正确统计：【切换动作次数】（从一个目标切到另一个目标才算）")
    print("=" * 140)

    total_actions = 0

    for run_id in failed_ids:
        f_m = pathlib.Path(f"{file_multi}_{run_id}.json")
        if not f_m.exists():
            continue

        with open(f_m, encoding='utf-8') as f:
            data_m = json.load(f)

        prev_idx = None
        action_count = 0

        for step in data_m:
            current_idx = str(step.get("current_target_index", "0"))

            # 第一次
            if prev_idx is None:
                prev_idx = current_idx
                continue

            # 目标变了 → 才算一次切换动作
            if current_idx != prev_idx:
                action_count += 1
                prev_idx = current_idx

        total_actions += action_count
        print(f"📌 ID {run_id:>2} | 真实切换动作：{action_count:>2} 次")

    print("\n" + "=" * 100)
    print(f"✅ 所有失败ID 总切换动作：{total_actions} 次")
    print("=" * 100)

# ===================== ✅ 检查：是否正确切换 & 是否成功（返回统计结果）=====================
def check_multi_target_switch(file_multi, common_ids):
    print("\n" + "=" * 140)
    print("🔎 专用检查：所有样本是否【真正切换目标】& 切换后是否成功")
    print("=" * 140)

    total_checked = 0
    has_switch = 0
    no_switch = 0
    switch_succ = 0
    switch_fail = 0
    noswitch_succ = 0
    noswitch_fail = 0

    switch_succ_ids = []
    switch_fail_ids = []
    noswitch_succ_ids = []
    noswitch_fail_ids = []

    for run_id in common_ids:
        f_m = pathlib.Path(f"{file_multi}_{run_id}.json")
        if not f_m.exists():
            continue

        total_checked += 1
        with open(f_m, 'r', encoding='utf-8') as f:
            data_m = json.load(f)

        switched_cnt = sum(1 for step in data_m if step.get("current_target_index", "0") != "0")
        is_switched = switched_cnt > 0
        final_succ = data_m[-1]["attack_success"]

        if is_switched:
            has_switch += 1
            if final_succ:
                switch_succ += 1
                switch_succ_ids.append(run_id)
            else:
                switch_fail += 1
                switch_fail_ids.append(run_id)
        else:
            no_switch += 1
            if final_succ:
                noswitch_succ += 1
                noswitch_succ_ids.append(run_id)
            else:
                noswitch_fail += 1
                noswitch_fail_ids.append(run_id)

    print(f"\n📌 总检查样本数：{total_checked}")
    print(f"\n【1】真正启用多目标（切换过目标）：{has_switch} 个")
    print(f"   ✅ 切换后成功：{switch_succ} 个  ID：{switch_succ_ids}")
    print(f"   ❌ 切换后失败：{switch_fail} 个  ID：{switch_fail_ids}")

    print(f"\n【2】未启用多目标（全程没切换）：{no_switch} 个")
    print(f"   ✅ 没切换也成功：{noswitch_succ} 个  ID：{noswitch_succ_ids}")
    print(f"   ❌ 没切换也失败：{noswitch_fail} 个  ID：{noswitch_fail_ids}")

    print("\n" + "=" * 140)
    print("🎯 多目标机制有效性总结")
    print("=" * 140)
    if has_switch == 0:
        print("❌ 所有样本都没有切换目标 → 多目标功能完全没生效！")
    else:
        if switch_succ > 0:
            print(f"✅ 多目标有效！{switch_succ} 个样本靠切换目标成功")
        if switch_fail > 0:
            print(f"⚠️ {switch_fail} 个样本切换目标后失败")
        if switch_succ + switch_fail > 0:
            succ_rate = switch_succ / (switch_succ + switch_fail) * 100
            print(f"\n📊 切换目标后的成功率：{succ_rate:.1f}%")
    print("=" * 140)

    # ✅ 自动返回【切换失败的ID列表】给后面用
    return switch_fail_ids



# ===================== 2. 核心分析函数 =====================
def analyze_results(file_prefix, method_name, common_ids):
    success_count = 0
    success_steps = []
    all_sample_steps = []

    success_cosine = []
    success_ppl = []
    all_final_cosine = []
    all_final_ppl = []

    print(f"\n正在分析 {method_name} ...")
    exist_count = len(common_ids)
    print(f"→ 共用 {exist_count} 个样本")

    for run_id in common_ids:
        file_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, list) or len(data) == 0:
                continue

            is_success = False
            first_success_step = None
            for step_data in data:
                if step_data.get('attack_success', False):
                    is_success = True
                    first_success_step = step_data.get('step', 0)
                    break

            final_step = data[-1].get('step', 0)
            final_cosine = data[-1].get('current_cosine_sim', 0.0)
            final_ppl = data[-1].get('ppl', 0.0)

            all_final_cosine.append(final_cosine)
            all_final_ppl.append(final_ppl)

            if is_success:
                success_count += 1
                success_steps.append(first_success_step)
                all_sample_steps.append(first_success_step)
                success_cosine.append(final_cosine)
                success_ppl.append(final_ppl)
            else:
                all_sample_steps.append(final_step)

        except Exception as e:
            print(f"[异常] {file_path.name}: {str(e)}")

    success_rate = (success_count / exist_count * 100) if exist_count > 0 else 0.0
    avg_steps_success = sum(success_steps) / len(success_steps) if len(success_steps) > 0 else 0.0
    avg_steps_total = sum(all_sample_steps) / len(all_sample_steps) if len(all_sample_steps) > 0 else 0.0

    avg_success_cosine = sum(success_cosine) / len(success_cosine) if success_cosine else 0.0
    avg_success_ppl = sum(success_ppl) / len(success_ppl) if success_ppl else 0.0
    avg_all_cosine = sum(all_final_cosine) / len(all_final_cosine) if all_final_cosine else 0.0
    avg_all_ppl = sum(all_final_ppl) / len(all_final_ppl) if all_final_ppl else 0.0

    fail_count = exist_count - success_count

    return (
        method_name, exist_count, success_count, fail_count, success_rate,
        avg_steps_success, avg_steps_total,
        avg_success_cosine, avg_success_ppl,
        avg_all_cosine, avg_all_ppl
    )


def analyze_diverge_by_id(target_id, file_single, file_multi, ce_thresh=0.1):
    print("\n" + "=" * 140)
    print(f"🎯 深度分析 ID = {target_id} | 单目标 ↔ 多目标 差异对比")
    print("=" * 140)

    f_s = pathlib.Path(f"{file_single}_{target_id}.json")
    f_m = pathlib.Path(f"{file_multi}_{target_id}.json")

    with open(f_s, encoding='utf-8') as f:
        log_s = json.load(f)
    with open(f_m, encoding='utf-8') as f:
        log_m = json.load(f)

    # 最终状态
    succ_s = log_s[-1]["attack_success"]
    succ_m = log_m[-1]["attack_success"]

    # 切换统计（真正判断是否启用了多目标）
    switch_times = sum(1 for step in log_m if step.get("current_target_index", "0") != "0")
    total_steps = len(log_m)
    switch_ratio = switch_times / total_steps * 100 if total_steps > 0 else 0
    is_multi_effective = switch_times > 0  # 🔥 只有切换了，多目标才算真正生效

    # 显示样本状态
    if succ_s and not succ_m:
        print(f"📌 状态：单目标成功 ✅ | 多目标失败 ❌（多目标变差）")
    elif not succ_s and succ_m:
        print(f"📌 状态：单目标失败 ❌ | 多目标成功 ✅（多目标提升）")
    else:
        pass
       # print(f"📌 状态：无显著差异")

    print(f"🔄 多目标是否真实启用：{'✅ 是（发生过切换）' if is_multi_effective else '❌ 否（全程单目标）'}")
    print(f"🔄 切换统计：总步={total_steps} | 切换次数={switch_times} | 切换比例={switch_ratio:.1f}%")
    print("-" * 140)

    # ===================== ✅ 真实结论判断 =====================
    print("\n🔥 真实原因（严格区分：多目标机制 / 随机波动）")

    # ----------------------
    # 情况1：根本没切换 = 多目标没起作用
    # ----------------------
    if not is_multi_effective:
        print(f"⚠️  关键结论：多目标【全程未切换目标】，等于没开多目标！")
        print(f"⚠️  本次差异 = 随机波动/优化不稳定，**不是多目标机制的效果**")

    # ----------------------
    # 情况2：真正用了多目标
    # ----------------------
    else:
        if succ_s and not succ_m:
            print(f"❌ 真实结论：多目标机制【失效】，切换目标导致优化崩溃")
        elif not succ_s and succ_m:
            print(f"✅ 真实结论：多目标机制【有效】，切换目标带来正向提升")

    print("=" * 140)

# ===================== 3. 主函数 =====================
def main():
    print("=" * 70)
    print("GCG 对抗攻击结果自动评估脚本 + 差异深度分析")
    print("=" * 70)

    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(r"D:\GitHub\I-GCG\Llama-2-7b-chat-hf_result\ours\20260426-034313\log")

    method_configs = [
        {"name": "single",       "con_loss": "", "mu": "",      "loss_type": "cross_entropy",   "sample_method": "", "target_similar_key":""},
        {"name": "similar1",     "con_loss": "", "mu": "multi", "loss_type": "cross_entropy",   "sample_method": "", "target_similar_key":"target_similar"},
    ]

    # ============== 获取共同ID ==============
    id_sets = []
    file_prefix_map = {}
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{cfg["sample_method"]}_{cfg["target_similar_key"]}'
        file_prefix_map[cfg["name"]] = file_prefix
        ids = set()
        for run_id in range(1, 50):
            f_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            if f_path.exists():
                ids.add(run_id)
        id_sets.append(ids)

    common_ids = sorted(set.intersection(*id_sets))
    print(f"\n✅ 共用样本ID: {common_ids}")
    print(f"✅ 共 {len(common_ids)} 个")

    # ============== 评估 ==============
    results = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{cfg["sample_method"]}_{cfg["target_similar_key"]}'
        res = analyze_results(file_prefix, cfg["name"], common_ids)
        results.append(res)

    # ===================== 输出表格 =====================
    print("\n" + "=" * 140)
    header = (
        f"{'方法':<26} | {'存在':<4} | {'成功':<4} | {'失败':<4} | {'成功率':>8} | "
        f"{'成功平均步数':>10} | {'全体平均步数':>10} | {'成功Cos':>10} | {'成功PPL':>10} | {'全体Cos':>10} | {'全体PPL':>10}"
    )
    print(header)
    print("-" * 140)
    for res in results:
        method, exist, success, fail, rate, avg_step_success, avg_step_total, \
        avg_success_cosine, avg_success_ppl, avg_all_cosine, avg_all_ppl = res
        print(
            f"{method:<26} | {exist:<4} | {success:<4} | {fail:<4} | {rate:>7.2f}% | "
            f"{avg_step_success:>9.2f} | {avg_step_total:>9.2f} | {avg_success_cosine:>9.3f} | {avg_success_ppl:>9.2f} | {avg_all_cosine:>9.3f} | {avg_all_ppl:>9.2f}"
        )
    print("=" * 140)

    # ===================== 结论 =====================
    print("\n📊 结论：")
    if len(results) < 2:
        print("→ 对比方法不足")
        return

    res1, res2 = results[0], results[1]
    _, exist1, suc1, _, rate1, step1, total1, cos1, ppl1, all_cos1, all_ppl1 = res1
    _, exist2, suc2, _, rate2, step2, total2, cos2, ppl2, all_cos2, all_ppl2 = res2

    diff_rate = rate2 - rate1
    if diff_rate > 0:
        print(f"→ {res2[0]} 成功率 +{diff_rate:.2f}%")
    elif diff_rate < 0:
        print(f"→ {res2[0]} 成功率 -{abs(diff_rate):.2f}%")
    else:
        print("→ 成功率相同")

    if suc1 > 0 and suc2 > 0:
        diff_step = step2 - step1
        if diff_step < 0:
            print(f"→ {res2[0]}【成功】平均快 {-diff_step:.2f} 步")
        else:
            print(f"→ {res2[0]}【成功】平均慢 {diff_step:.2f} 步")
        diff_cos = cos2 - cos1
        print(f"→ {res2[0]}【成功余弦相似度】变化：{diff_cos:.3f}")
        diff_ppl = ppl2 - ppl1
        print(f"→ {res2[0]}【成功PPL】变化：{diff_ppl:.2f}")

    diff_total = total2 - total1
    if diff_total < 0:
        print(f"→ {res2[0]}【全体】平均快 {-diff_total:.2f} 步")
    else:
        print(f"→ {res2[0]}【全体】平均慢 {diff_total:.2f} 步")

    # ===================== ✅ 逐个分析所有存在差异的 ID =====================
    # ===================== ✅ 全面差异分析：单目标 vs 多目标（按 mu 字段区分） =====================


    print("\n" + "=" * 120)
    print("🔍 全面差异分析 | 单目标(mu='') VS 多目标(mu='multi') | 不漏任何情况")
    print("=" * 120)

    # 自动从配置中匹配单/多目标（绝对正确，不看name）
    file_single = None
    file_multi = None
    for cfg in method_configs:
        fp = file_prefix_map[cfg["name"]]
        if cfg["mu"] == "":
            file_single = fp
        elif cfg["mu"] == "multi":
            file_multi = fp

    # 全局统计
    total_ids = 0
    all_succ = 0
    all_fail = 0
    single_succ_multi_fail = 0  # 多目标变差
    single_fail_multi_succ = 0  # 多目标变好

    for run_id in common_ids:
        f_s = pathlib.Path(f"{file_single}_{run_id}.json")
        f_m = pathlib.Path(f"{file_multi}_{run_id}.json")
        if not (f_s.exists() and f_m.exists()):
            continue

        total_ids += 1
        with open(f_s, 'r', encoding='utf-8') as f:
            data_s = json.load(f)
        with open(f_m, 'r', encoding='utf-8') as f:
            data_m = json.load(f)

        succ_s = data_s[-1]["attack_success"]
        succ_m = data_m[-1]["attack_success"]

        # 清晰输出每一个样本
        print(f"\n📌 ID {run_id} | 单目标：{'✅成功' if succ_s else '❌失败'} | 多目标：{'✅成功' if succ_m else '❌失败'}")

        # 分类统计
        if succ_s and succ_m:
            all_succ += 1
            #print("   → 全部成功，无差异")
        elif not succ_s and not succ_m:
            all_fail += 1
            #print("   → 全部失败，无差异")
        elif succ_s and not succ_m:
            single_succ_multi_fail += 1
            #print("   ⚠️  差异：单成功 → 多失败 | 多目标变差")
            #analyze_diverge_by_id(run_id, file_single, file_multi)  # 有分析
        elif not succ_s and succ_m:
            single_fail_multi_succ += 1
            print("   🚀 差异：单失败 → 多成功 | 多目标提升！")
            analyze_diverge_by_id(run_id, file_single, file_multi)  # ✅ 现在也有深度分析！
    # ===================== 📊 最终汇总（可直接写论文） =====================
    print("\n" + "=" * 120)
    print("📊 多目标攻击效果 完整统计")
    print("=" * 120)
    print(f"总样本数：{total_ids}")
    print(f"全部成功：{all_succ}")
    print(f"全部失败：{all_fail}")
    print(f"⚠️  单成功 多失败（变差）：{single_succ_multi_fail}")
    print(f"🚀 单失败 多成功（提升）：{single_fail_multi_succ}")

    print("\n💡 核心结论：")
    if single_succ_multi_fail > single_fail_multi_succ:
        print("   多目标整体导致性能下降，多数样本变差")
    elif single_fail_multi_succ > single_succ_multi_fail:
        print("   多目标有效提升了成功率，具有增益效果")
    else:
        print("   多目标无明显正向/负向影响，效果持平")
    print("=" * 120)

    # ===================== 调用 =====================
    # 1. 运行检查，自动获取 switch_fail_ids
    failed_ids = check_multi_target_switch(file_multi, common_ids)

    # 2. 自动统计这些ID的总切换次数（不用手动输入ID）
   # count_total_switch_times(file_multi, failed_ids)
    count_real_switch_actions(file_multi, failed_ids)



if __name__ == "__main__":
    main()