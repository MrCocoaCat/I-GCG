import json
import pathlib
import argparse
import datetime

# ===================== 1. 修复版配置 =====================
parser = argparse.ArgumentParser(description="")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260403-195405')

parser.add_argument('--batch_size', type=int, default=6)
parser.add_argument('--top_k', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=500)
# 🔥 修复：bool 写法错误
parser.add_argument('--early_stop', type=lambda x: x.lower() == 'true', default=True)
parser.add_argument('--loss_type', type=str, default="cross_entropy", choices=["cross_entropy", "cosine"])
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--str_init', type=str, default="adv_init_suffix")
parser.add_argument('--use_multi_target', type=lambda x: x.lower() == 'true', default=True)

args = parser.parse_args()


# ===================== 2. 核心分析函数（仅统计存在文件） =====================
def analyze_results(file_prefix, method_name, max_possible_id=200):
    exist_ids = []
    success_count = 0
    success_steps = []

    print(f"\n正在分析 {method_name} ...")

    # 自动探测存在的 ID
    for run_id in range(1, max_possible_id + 1):
        file_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
        if file_path.exists():
            exist_ids.append(run_id)

    if not exist_ids:
        print(f"→ 无任何文件存在")
        return 0, 0, 0, 0.0, 0.0

    exist_count = len(exist_ids)
    print(f"→ 存在文件 ID: {sorted(exist_ids)}")
    print(f"→ 共 {exist_count} 个有效样本")

    # 只读取存在的文件
    for run_id in exist_ids:
        file_path = pathlib.Path(f"{file_prefix}_{run_id}.json")
        print(f"[DEBUG] 读取：{file_path.resolve()}")

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

            if not is_success and len(data) > 0:
                final_step = data[-1]
                is_success = final_step.get('attack_success', False)
                if is_success:
                    first_success_step = final_step.get('step', 0)

            if is_success:
                success_count += 1
                if first_success_step is not None:
                    success_steps.append(first_success_step)

        except Exception as e:
            print(f"[异常] {file_path.name}: {str(e)}")

    success_rate = (success_count / exist_count * 100) if exist_count > 0 else 0.0
    avg_steps = sum(success_steps) / len(success_steps) if len(success_steps) > 0 else 0.0

    return exist_count, success_count, exist_count - success_count, success_rate, avg_steps


# ===================== 3. 主函数 =====================
def main():
    print("=" * 60)
    print("GCG 对抗攻击结果自动评估脚本（仅统计存在样本）")
    print("=" * 60)

    mu = ""
    ppl_suffix = "_ppl" if args.use_ppl_filter else ""

    single_file_prefix = pathlib.Path(
        f'{args.output_path}/log/result_{mu}{args.loss_type}{ppl_suffix}_{args.str_init}'
    )
    print(f"单目标前缀：{single_file_prefix}")

    mu = "multi_target"
    multi_file_prefix = pathlib.Path(
        f'{args.output_path}/log/result_{mu}{args.loss_type}{ppl_suffix}_{args.str_init}'
    )
    print(f"多目标前缀：{multi_file_prefix}")

    # 分析
    exist_s, success_s, fail_s, rate_s, avg_steps_s = analyze_results(single_file_prefix, "单目标攻击")
    exist_m, success_m, fail_m, rate_m, avg_steps_m = analyze_results(multi_file_prefix, "多目标攻击")

    # 输出表格
    print("\n" + "=" * 90)
    print(f"{'方法':<20} | {'存在':<6} | {'成功':<6} | {'失败':<6} | {'成功率':<10} | {'平均步数'}")
    print("-" * 90)
    print(f"{'单目标':<20} | {exist_s:<6} | {success_s:<6} | {fail_s:<6} | {rate_s:>8.2f}% | {avg_steps_s:>10.2f}")
    print(f"{'多目标':<20} | {exist_m:<6} | {success_m:<6} | {fail_m:<6} | {rate_m:>8.2f}% | {avg_steps_m:>10.2f}")
    print("=" * 90)

    # 结论
    print("\n📊 结论：")
    if exist_s == 0 or exist_m == 0:
        print("→ 有效样本不足")
        return

    diff_rate = rate_m - rate_s
    if diff_rate > 0:
        print(f"→ 多目标 成功率 +{diff_rate:.2f}%")
    elif diff_rate < 0:
        print(f"→ 多目标 成功率 -{abs(diff_rate):.2f}%")
    else:
        print("→ 成功率相同")

    if success_s > 0 and success_m > 0:
        diff_step = avg_steps_m - avg_steps_s
        if diff_step < 0:
            print(f"→ 多目标 平均快 {-diff_step:.2f} 步")
        else:
            print(f"→ 多目标 平均慢 {diff_step:.2f} 步")

    print("=" * 90)


if __name__ == "__main__":
    main()