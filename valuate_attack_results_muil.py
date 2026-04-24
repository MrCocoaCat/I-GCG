import json
import pathlib
import argparse

# ===================== 1. 配置解析 =====================
parser = argparse.ArgumentParser(description="GCG 攻击结果自动评估脚本")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260415-014710-MulMethodSucess')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--loss_type', type=str, default="cross_entropy", choices=["cross_entropy", "cosine"])
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--str_init', type=str, default="adv_init_suffix")
parser.add_argument('--use_multi_target', type=lambda x: x.lower() == 'true', default=True)

args = parser.parse_args()


# ===================== 核心：仅根据 current_target_index 变化判定多目标是否生效 =====================
def analyze_inconsistency_cause(inc_data):
    names = inc_data["method_names"]
    name_si, name_multi = names[0], names[1]
    log_data = inc_data["log_data"]

    # 工具函数：检查【多目标样本】是否切换过 target index（0→非0）
    def check_target_switched(sid):
        data = log_data[name_multi].get(sid, [])
        if not data:
            return False, 0, []

        target_indices = []
        switched = False
        for step in data:
            idx = step.get("current_target_index", "0")
            target_indices.append(idx)
            if idx != "0":
                switched = True
        return switched, len([x for x in target_indices if x != "0"]), target_indices

    print("\n" + "=" * 70)
    print("🎯 多目标生效判定（仅依据 current_target_index 是否变化）")
    print("=" * 70)

    # 统计所有不一致样本
    inconsistent_ids = inc_data["si_succ_mult_fail"] + inc_data["si_fail_mult_succ"]

    if not inconsistent_ids:
        print("✅ 无不一致样本，多目标效果稳定")
        return

    print(f"📌 不一致样本总数：{len(inconsistent_ids)}")
    print("🔎 逐样本判定多目标是否真实生效：")
    print("-" * 70)
    print(f"{'样本ID':<6} {'目标是否切换':<10} {'切换次数':<8} {'判定结果':<20}")
    print("-" * 70)

    # 逐样本分析
    for sid in sorted(inconsistent_ids):
        switched, switch_cnt, indices = check_target_switched(sid)

        # 判定结果
        if switched:
            judge = "🚀 多目标真实生效"
        else:
            judge = "❌ 多目标未生效(纯随机)"

        print(f"{sid:<6} {str(switched):<10} {switch_cnt:<8} {judge:<20}")

    print("-" * 70)
    print("\n📢 最终判定规则（你定义的）：")
    print("1. current_target_index 从 0 变成其他数字 = 多目标【真的生效】")
    print("2. current_target_index 全程 = 0 = 多目标【未生效】，和单目标无区别")
    print("=" * 70)

    # 汇总不一致原因
    switched_ids = []
    random_ids = []
    for sid in inconsistent_ids:
        switched, _, _ = check_target_switched(sid)
        if switched:
            switched_ids.append(sid)
        else:
            random_ids.append(sid)

    print("\n📊 不一致原因汇总：")
    print(f"✅ 多目标生效导致不一致：{len(switched_ids)} 个样本 {switched_ids}")
    print(f"❌ 纯随机导致不一致：{len(random_ids)} 个样本 {random_ids}")
    print("=" * 70)
# ===================== 2. 核心分析函数 =====================
def analyze_results(file_prefix, method_name, common_ids):
    success_count = 0
    success_steps = []
    all_sample_steps = []

    # ======== 新增：cosine_sim & ppl ========
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

            # === 1. 找第一次成功步数 ===
            is_success = False
            first_success_step = None
            for step_data in data:
                if step_data.get('attack_success', False):
                    is_success = True
                    first_success_step = step_data.get('step', 0)
                    break

            # === 2. 最终步 ===
            final_step = data[-1].get('step', 0)
            final_cosine = data[-1].get('current_cosine_sim', 0.0)
            final_ppl = data[-1].get('ppl', 0.0)

            all_final_cosine.append(final_cosine)
            all_final_ppl.append(final_ppl)

            # === 3. 统计 ===
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

    # ======== 新增指标计算 ========
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

# ===================== 检测不一致样本（返回详细数据） =====================
def find_inconsistent_samples(method_configs, log_dir, ppl_suffix, args, common_ids):
    # 存储每个方法：样本ID -> 是否成功
    method_status = {}
    # 存储每个方法：样本ID -> 完整日志数据
    method_log_data = {}

    for cfg in method_configs:
        name = cfg["name"]
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        status = {}
        log_data = {}

        for run_id in common_ids:
            file_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                log_data[run_id] = data
                # 判断是否成功
                is_success = any(step.get('attack_success', False) for step in data)
                status[run_id] = is_success
            except Exception as e:
                status[run_id] = False
                log_data[run_id] = []

        method_status[name] = status
        method_log_data[name] = log_data

    # 拆分四类样本
    name1, name2 = method_configs[0]["name"], method_configs[1]["name"]
    s1, s2 = method_status[name1], method_status[name2]

    both_success = []    # 都成功
    both_fail = []       # 都失败
    s1_succ_s2_fail = [] # 单成功、多失败
    s1_fail_s2_succ = [] # 单失败、多成功

    for sid in common_ids:
        a, b = s1.get(sid, False), s2.get(sid, False)
        if a and b:
            both_success.append(sid)
        elif not a and not b:
            both_fail.append(sid)
        elif a and not b:
            s1_succ_s2_fail.append(sid)
        else:
            s1_fail_s2_succ.append(sid)

    # 返回：分类样本 + 状态 + 原始日志
    return {
        "both_success": both_success,
        "both_fail": both_fail,
        "si_succ_mult_fail": s1_succ_s2_fail,
        "si_fail_mult_succ": s1_fail_s2_succ,
        "status": method_status,
        "log_data": method_log_data,
        "method_names": (name1, name2)
    }

# ===================== 核心：分析不一致原因（随机 vs 多目标生效） =====================

# ===================== 3. 主函数 =====================
def main():
    print("=" * 70)
    print("GCG 对抗攻击结果自动评估脚本")
    print("=" * 70)

    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    # ppl_suffix = "ppl" if args.use_ppl_filter else ""
    #     mu = "multi" if args.use_multi_target else ""
    #     con_loss = "contrast" if args.use_contrast_loss else ""
    #     sample_method = "weighted_sample" if args.use_weighted_sample else ""
    method_configs = [
        {"name": "si",    "con_loss": "", "mu": "",      "loss_type": "cross_entropy",   "sample_method": ""},
        {"name": "mult", "con_loss": "", "mu": "multi", "loss_type": "cross_entropy",   "sample_method": ""},

    ]
# __cross_entropy__adv_init_suffix__1.json

    # D:\GitHub\I-GCG\test_select_method\ours\20260419-213411\log
    # ============== 关键：获取所有方法共同存在的样本 ID ==============
    id_sets = []
    for cfg in method_configs:
        #log_json_file = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}_{cfg["sample_method"]}_{cfg["target_similar_key"]}'
        file_prefix =   log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
       # log_json_file = {mu}_{con_loss}_{args.loss_type}_{ppl_suffix}_{args.str_init}_{sample_method}_{args.id}.json')
        ids = set()
        for run_id in range(1, 50):

            f_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            if f_path.exists():
                ids.add(run_id)
            else:
               print(f_path)
        id_sets.append(ids)

    common_ids = sorted(set.intersection(*id_sets))
    print(f"\n✅ 所有方法共用样本 ID: {common_ids}")
    print(f"✅ 共 {len(common_ids)} 个对比样本")



    # ============== 统一用 common_ids 评估 ==============
    results = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        # file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}_{cfg["sample_method"]}'
        res = analyze_results(file_prefix, cfg["name"], common_ids)
        results.append(res)

    # ===================== 输出表格 =====================
    print("\n" + "=" * 140)
    header = (
        f"{'方法':<26} | "
        f"{'存在':<4} | "
        f"{'成功':<4} | "
        f"{'失败':<4} | "
        f"{'成功率':>8} | "
        f"{'成功平均步数':>10} | "
        f"{'全体平均步数':>10} | "
        f"{'成功Cos':>10} | "
        f"{'成功PPL':>10} | "
        f"{'全体Cos':>10} | "
        f"{'全体PPL':>10}"
    )
    print(header)
    print("-" * 140)

    for res in results:
        method, exist, success, fail, rate, avg_step_success, avg_step_total, \
        avg_success_cosine, avg_success_ppl, avg_all_cosine, avg_all_ppl = res

        print(
            f"{method:<26} | "
            f"{exist:<4} | "
            f"{success:<4} | "
            f"{fail:<4} | "
            f"{rate:>7.2f}% | "
            f"{avg_step_success:>9.2f} | "
            f"{avg_step_total:>9.2f} | "
            f"{avg_success_cosine:>9.3f} | "
            f"{avg_success_ppl:>9.2f} | "
            f"{avg_all_cosine:>9.3f} | "
            f"{avg_all_ppl:>9.2f}"
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
            print(f"→ {res2[0]}【成功样本】平均快 {-diff_step:.2f} 步")
        else:
            print(f"→ {res2[0]}【成功样本】平均慢 {diff_step:.2f} 步")

        diff_cos = cos2 - cos1
        print(f"→ {res2[0]}【成功余弦相似度】变化：{diff_cos:.3f}")

        diff_ppl = ppl2 - ppl1
        print(f"→ {res2[0]}【成功困惑度PPL】变化：{diff_ppl:.2f}")

    diff_total = total2 - total1
    if diff_total < 0:
        print(f"→ {res2[0]}【全体样本】平均快 {-diff_total:.2f} 步")
    else:
        print(f"→ {res2[0]}【全体样本】平均慢 {diff_total:.2f} 步")

    print("=" * 100)
    #find_inconsistent_samples(method_configs, log_dir, ppl_suffix, args, common_ids)

    # 新增：检测不一致样本 + 分析原因
    inc_data = find_inconsistent_samples(method_configs, log_dir, ppl_suffix, args, common_ids)

    analyze_inconsistency_cause(inc_data)

    #analyze_inconsistency_cause(inc_data, method_configs, log_dir, ppl_suffix, args)


if __name__ == "__main__":
    main()