import json
import pathlib
import argparse

# ===================== 1. 配置解析 =====================
parser = argparse.ArgumentParser(description="GCG 攻击结果自动评估脚本")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260411-044121')
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

            # === 3. 统计 ===
            if is_success:
                success_count += 1
                success_steps.append(first_success_step)
                all_sample_steps.append(first_success_step)
            else:
                all_sample_steps.append(final_step)

        except Exception as e:
            print(f"[异常] {file_path.name}: {str(e)}")

    success_rate = (success_count / exist_count * 100) if exist_count > 0 else 0.0
    avg_steps_success = sum(success_steps) / len(success_steps) if len(success_steps) > 0 else 0.0
    avg_steps_total = sum(all_sample_steps) / len(all_sample_steps) if len(all_sample_steps) > 0 else 0.0
    fail_count = exist_count - success_count

    return (method_name, exist_count, success_count, fail_count, success_rate, avg_steps_success, avg_steps_total)


# ===================== 3. 主函数 =====================
def main():
    print("=" * 70)
    print("GCG 对抗攻击结果自动评估脚本（统一样本公平对比）")
    print("=" * 70)

    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    method_configs = [
        {"name": "单目标 - Contrast Loss",    "con_loss": "contrast", "mu": ""    , "loss_type": "cross_entropy",},
        {"name": "单目标 - No Contrast Loss", "con_loss": "",         "mu": "",     "loss_type": "cross_entropy"},
        {"name": "单目标 - loss type Contrast   ", "con_loss": "",  "mu": ""    , "loss_type": "contrast"},
        {"name": "多目标 - Contrast Loss", "con_loss": "contrast", "mu": "multi", "loss_type": "cross_entropy"},
        {"name": "多目标 - No Contrast Loss", "con_loss": "",         "mu": "multi","loss_type": "cross_entropy"}
    ]

    # ============== 关键：获取所有方法共同存在的样本 ID ==============
    id_sets = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{cfg["loss_type"]}_{ppl_suffix}_{args.str_init}'
        ids = set()
        for run_id in range(1, 200):
            f_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
            if f_path.exists():
                ids.add(run_id)
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
    print("\n" + "=" * 100)
    print(f"{'方法':<22} | {'存在':<5} | {'成功':<5} | {'失败':<5} | {'成功率':<9} | {'成功平均步数':<10} | {'全体平均步数'}")
    print("-" * 100)

    for res in results:
        method, exist, success, fail, rate, avg_step_success, avg_step_total = res
        print(f"{method:<22} | {exist:<5} | {success:<5} | {fail:<5} | {rate:>7.2f}% | {avg_step_success:>9.2f} | {avg_step_total:>11.2f}")

    print("=" * 100)

    # ===================== 结论 =====================
    print("\n📊 结论：")
    if len(results) < 2:
        print("→ 对比方法不足")
        return

    res1, res2 = results[0], results[1]
    _, exist1, suc1, _, rate1, step1, total1 = res1
    _, exist2, suc2, _, rate2, step2, total2 = res2

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

    diff_total = total2 - total1
    if diff_total < 0:
        print(f"→ {res2[0]}【全体样本】平均快 {-diff_total:.2f} 步")
    else:
        print(f"→ {res2[0]}【全体样本】平均慢 {diff_total:.2f} 步")

    print("=" * 100)


if __name__ == "__main__":
    main()