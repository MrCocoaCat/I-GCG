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


# ===================== 3. 主函数 =====================
def main():
    print("=" * 70)
    print("GCG 对抗攻击结果自动评估脚本（统一样本公平对比）")
    print("=" * 70)

    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    method_configs = [
        {"name": "单目标 - No Contrast Loss", "con_loss": "", "mu": "", "loss_type": "cross_entropy" ,  "    sample_method" :""},
        {"name": "多目标 - No Contrast Loss", "con_loss": "", "mu": "multi", "loss_type": "cross_entropy", "sample_method": ""}
    ]

    # ============== 关键：获取所有方法共同存在的样本 ID ==============
    id_sets = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        ids = set()
        for run_id in range(1, 50):
            f_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            if f_path.exists():
                ids.add(run_id)
           # else:
               # print(f_path)
        id_sets.append(ids)

    common_ids = sorted(set.intersection(*id_sets))
    print(f"\n✅ 所有方法共用样本 ID: {common_ids}")
    print(f"✅ 共 {len(common_ids)} 个对比样本")

    # ============== 统一用 common_ids 评估 ==============
    results = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{args.loss_type}_{ppl_suffix}_{args.str_init}'
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


if __name__ == "__main__":
    main()