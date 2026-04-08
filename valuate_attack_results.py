import json
import pathlib
import argparse
import datetime

# ===================== 1. 配置解析 =====================
parser = argparse.ArgumentParser(description="GCG 攻击结果自动评估脚本")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260408-021000')
parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--loss_type', type=str, default="cross_entropy", choices=["cross_entropy", "cosine"])
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--str_init', type=str, default="adv_init_suffix")
parser.add_argument('--use_multi_target', type=lambda x: x.lower() == 'true', default=True)

args = parser.parse_args()


# ===================== 2. 核心分析函数 =====================
def analyze_results(file_prefix, method_name, max_possible_id=200):
    exist_ids = []
    success_count = 0
    success_steps = []       # 只存成功的步数
    all_sample_steps = []   # 所有样本的步数（成功+失败）

    print(f"\n正在分析 {method_name} ...")

    for run_id in range(1, max_possible_id + 1):
        file_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
        if file_path.exists():
            exist_ids.append(run_id)

    if not exist_ids:
        print(f"→ 无任何文件存在")
        return (method_name, 0, 0, 0, 0.0, 0.0, 0.0)

    exist_count = len(exist_ids)
    print(f"→ 存在文件 ID: {sorted(exist_ids)}")
    print(f"→ 共 {exist_count} 个有效样本")

    for run_id in exist_ids:
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

            # === 2. 最终步检查 ===
            final_step = data[-1].get('step', 0)
            if not is_success:
                final_step = data[-1].get('step', 0)

            # === 3. 统计：所有样本都计入总平均步数 ===
            if is_success:
                success_count += 1
                success_steps.append(first_success_step)
                all_sample_steps.append(first_success_step)  # 成功计入
            else:
                all_sample_steps.append(final_step)         # 失败也计入（用总步数）

        except Exception as e:
            print(f"[异常] {file_path.name}: {str(e)}")

    # 计算两种平均步数
    success_rate = (success_count / exist_count * 100) if exist_count > 0 else 0.0
    avg_steps_success = sum(success_steps) / len(success_steps) if len(success_steps) > 0 else 0.0
    avg_steps_total = sum(all_sample_steps) / len(all_sample_steps) if len(all_sample_steps) > 0 else 0.0
    fail_count = exist_count - success_count

    return (method_name, exist_count, success_count, fail_count, success_rate, avg_steps_success, avg_steps_total)


# ===================== 3. 主函数 =====================
def main():
    print("=" * 70)
    print("GCG 对抗攻击结果自动评估脚本（成功平均步数 + 全体平均步数）")
    print("=" * 70)

    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    method_configs = [
        {"name": "单目标 - Contrast Loss",    "con_loss": "contrast", "mu": ""},
        {"name": "多目标 - Contrast Loss",    "con_loss": "contrast", "mu": "multi"},
        {"name": "单目标 - No Contrast Loss", "con_loss": "",         "mu": ""},
        {"name": "多目标 - No Contrast Loss", "con_loss": "",         "mu": "multi"}
    ]

    results = []
    for cfg in method_configs:
        file_prefix = log_dir / f'{cfg["mu"]}_{cfg["con_loss"]}_{args.loss_type}_{ppl_suffix}_{args.str_init}'
        res = analyze_results(file_prefix, cfg["name"])
        results.append(res)

    # ===================== 输出表格（新增一列）=====================
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

    if exist1 == 0 or exist2 == 0:
        print("→ 有效样本不足")
        return

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