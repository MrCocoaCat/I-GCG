import json
import pathlib
import argparse
import numpy as np

# ===================== 1. 配置解析 =====================
parser = argparse.ArgumentParser(description="GCG 攻击结果自动评估脚本")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260423-224658')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--loss_type', type=str, default="cross_entropy", choices=["cross_entropy", "cosine"])
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--str_init', type=str, default="adv_init_suffix")
parser.add_argument('--use_multi_target', type=lambda x: x.lower() == 'true', default=True)

args = parser.parse_args()


# ===================== 2. 核心分析函数（和你旧版完全一样接口） =====================
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


# ===================== 3. 多目标深度诊断（新版超强功能） =====================
def diagnose_multi_target_issue(method_configs, log_dir, ppl_suffix, args, common_ids):
    print("\n" + "=" * 120)
    print("🔥  多目标 & 单目标 差异深度诊断（为什么成功率下降）")
    print("=" * 120)

    file_map = {}
    for cfg in method_configs:
        key = (cfg["mu"], cfg["loss_type"])
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}_{cfg["sample_method"]}_{cfg["target_similar_key"]}'
        file_map[key] = file_prefix

    file_single = file_map.get(("", "cross_entropy"))
    file_multi = file_map.get(("multi", "cross_entropy"))

    target_id = None
    for run_id in common_ids:
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
        print("❌ 未找到 单成功+多失败 的样本")
        return

    print(f"🎯 找到问题样本 ID = {target_id}")
    print("=" * 120)

    with open(f"{file_single}_{target_id}.json", encoding='utf-8') as f:
        log_s = json.load(f)
    with open(f"{file_multi}_{target_id}.json", encoding='utf-8') as f:
        log_m = json.load(f)

    s_final = log_s[-1]
    m_final = log_m[-1]

    print(f"📊 最终对比")
    print(f"单目标 | 成功={s_final['attack_success']} | CE={s_final['current_cross_entropy']:.3f} | Cos={s_final['current_cosine_sim']:.3f}")
    print(f"多目标 | 成功={m_final['attack_success']} | CE={m_final['current_cross_entropy']:.3f} | Cos={m_final['current_cosine_sim']:.3f}")
    print("-" * 120)

    switch_times = sum(1 for step in log_m if step.get("current_target_index", "0") != "0")
    total_steps = len(log_m)
    switch_ratio = switch_times / total_steps * 100

    print(f"🔄 多目标切换：总步数 {total_steps} | 切换 {switch_times} 次 | 比例 {switch_ratio:.1f}%")

    print("\n📌 差异原因：")
    if switch_ratio > 10:
        print("❌ 多目标频繁切换 → 优化方向混乱 → 攻击失败")
    if m_final['current_cross_entropy'] > s_final['current_cross_entropy'] * 2:
        print("❌ 多目标损失函数远高于单目标 → 未收敛")
    if m_final['current_cosine_sim'] < s_final['current_cosine_sim'] * 0.7:
        print("❌ 多目标相似度太低 → 匹配失败")

    print("=" * 120)
    return target_id


# ===================== 4. 主函数（和旧版完全一样） =====================
def main():
    print("=" * 70)
    print("GCG 对抗攻击结果评估 + 差异深度诊断")
    print("=" * 70)

    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    method_configs = [
        {"name": "singal",       "con_loss": "", "mu": "",      "loss_type": "cross_entropy",   "sample_method": "", "target_similar_key":""},
        {"name": "similar1", "con_loss": "", "mu": "multi", "loss_type": "cross_entropy",   "sample_method": "", "target_similar_key":"target_similar"},
    ]

    # ============== 找共用ID（和旧版完全一样） ==============
    id_sets = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}_{cfg["sample_method"]}_{cfg["target_similar_key"]}'
        ids = set()
        for run_id in range(1, 50):
            f_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            if f_path.exists():
                ids.add(run_id)
        id_sets.append(ids)

    common_ids = sorted(set.intersection(*id_sets))
    print(f"\n✅ 共用样本 ID: {common_ids}")
    print(f"✅ 共 {len(common_ids)} 个")

    # ============== 评估（和旧版完全一样） ==============
    results = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}_{cfg["sample_method"]}_{cfg["target_similar_key"]}'
        res = analyze_results(file_prefix, cfg["name"], common_ids)
        results.append(res)

    # ============== 输出表格（和旧版完全一样） ==============
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

    # ============== 新版：深度诊断为什么有差异 ==============
    diagnose_multi_target_issue(method_configs, log_dir, ppl_suffix, args, common_ids)

    # ============== 结论（和旧版一样） ==============
    print("\n📊 结论：")
    if len(results) >= 2:
        res1, res2 = results[0], results[1]
        _, exist1, suc1, _, rate1, step1, total1, cos1, ppl1, all_cos1, all_ppl1 = res1
        _, exist2, suc2, _, rate2, step2, total2, cos2, ppl2, all_cos2, all_ppl2 = res2

        diff_rate = rate2 - rate1
        if diff_rate != 0:
            print(f"→ {res2[0]} 成功率 {'+' if diff_rate>0 else ''}{diff_rate:.2f}%")

        if suc1 and suc2:
            diff_step = step2 - step1
            if diff_step != 0:
                print(f"→ {res2[0]} 平均{'快' if diff_step<0 else '慢'} {abs(diff_step):.2f} 步")

    print("=" * 100)


if __name__ == "__main__":
    main()