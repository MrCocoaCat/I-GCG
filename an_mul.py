import json
import pathlib
import argparse
import numpy as np

# ===================== 配置 =====================
parser = argparse.ArgumentParser(description="多目标效果差 深度诊断脚本")
parser.add_argument('--output_path', type=str, default=r'D:\GitHub\I-GCG\test_select_method\ours\20260408-021000')
parser.add_argument('--loss_type', type=str, default="cross_entropy")
parser.add_argument('--use_ppl_filter', type=lambda x: x.lower() == 'true', default=False)
parser.add_argument('--str_init', type=str, default="adv_init_suffix")
args = parser.parse_args()

# ===================== 核心诊断函数 =====================
def diagnose_multi_target_issue(file_prefix_single, file_prefix_multi, max_id=200):
    print("\n" + "="*80)
    print("🔥  多目标效果差 深度诊断报告（适配你的日志格式）")
    print("="*80)

    exist_ids = []
    for run_id in range(1, max_id+1):
        f1 = pathlib.Path(f"{file_prefix_single}_{run_id}.json")
        f2 = pathlib.Path(f"{file_prefix_multi}_{run_id}.json")
        if f1.exists() and f2.exists():
            exist_ids.append(run_id)

    if not exist_ids:
        print("❌ 没有找到匹配的单目标/多目标日志")
        return

    print(f"✅ 可对比样本数：{len(exist_ids)}")
    print("-"*80)

    total_steps_multi = 0
    total_stayed_on_main = 0  # target_type == 0 的次数
    total_switched = 0

    all_multi_ce = []
    all_single_ce = []
    all_multi_cos = []
    all_single_cos = []
    multi_success = 0
    single_success = 0

    for run_id in exist_ids:
        f_multi = pathlib.Path(f"{file_prefix_multi}_{run_id}.json")
        f_single = pathlib.Path(f"{file_prefix_single}_{run_id}.json")

        with open(f_multi, 'r', encoding='utf-8') as f1:
            data_multi = json.load(f1)
        with open(f_single, 'r', encoding='utf-8') as f2:
            data_single = json.load(f2)

        if len(data_multi) == 0 or len(data_single) == 0:
            print(f"🆔 ID {run_id:2d} | 日志为空，跳过")
            continue

        # ---------------------
        # 🔍 统计 target_type
        # ---------------------
        stayed_on_main = 0
        switched = 0
        for step in data_multi:
            t_str = step.get('current_target_index')
            t = int(t_str)
            if t == 0:
                stayed_on_main += 1
            else:
                switched += 1

        total_stayed_on_main += stayed_on_main
        total_switched += switched
        total_steps_multi += len(data_multi)

        # ---------------------
        # 最终指标
        # ---------------------
        final_multi = data_multi[-1]
        final_single = data_single[-1]

        ce_multi = final_multi.get("current_cross_entropy", 999)
        ce_single = final_single.get("current_cross_entropy", 999)
        cos_multi = final_multi.get("current_cosine_sim", -1)
        cos_single = final_multi.get("current_cosine_sim", -1)

        all_multi_ce.append(ce_multi)
        all_single_ce.append(ce_single)
        all_multi_cos.append(cos_multi)
        all_single_cos.append(cos_single)

        suc_multi = final_multi.get("attack_success", False)
        suc_single = final_single.get("attack_success", False)
        if suc_multi: multi_success +=1
        if suc_single: single_success +=1

        # ---------------------
        # 输出
        # ---------------------
        print(f"🆔 ID {run_id:2d} | 全程主目标:{stayed_on_main}/{len(data_multi)} | 切换:{switched}")
        print(f"        CE  单:{ce_single:6.3f}   多:{ce_multi:6.3f}   差:{ce_multi-ce_single:+.3f}")
        print(f"        成功  单:{suc_single!r:<5}    多:{suc_multi!r:<5}")
        print("-"*80)

    # ===================== 最终诊断 =====================
    print("\n" + "="*80)
    print("📌 最终诊断结论")
    print("="*80)

    total = len(exist_ids)
    stay_rate = total_stayed_on_main / total_steps_multi * 100
    print(f"🔍 多目标全程使用主目标(target_type=0)：{total_stayed_on_main}/{total_steps_multi} 步 ({stay_rate:.1f}%)")

    if stay_rate > 99:
        print("✅ **结论：多目标全程没有切换任何相似目标！完全等于单目标设置！**")
        print("⚠️ **但实际效果更差 → 说明代码内部有额外干扰！**")

    avg_ce_multi = np.mean(all_multi_ce)
    avg_ce_single = np.mean(all_single_ce)
    ce_diff = avg_ce_multi - avg_ce_single

    print(f"\n📉 平均 CE 损失：")
    print(f"   单目标：{avg_ce_single:.3f}")
    print(f"   多目标：{avg_ce_multi:.3f}")
    print(f"   差距：{ce_diff:+.3f}")

    s_rate = single_success / total * 100
    m_rate = multi_success / total * 100

    print(f"\n🎯 成功率：")
    print(f"   单目标：{s_rate:.1f}%")
    print(f"   多目标：{m_rate:.1f}%")
    print(f"   差距：{m_rate - s_rate:.1f}%")

    print("\n💡 病根定位：")
    if stay_rate > 99 and ce_diff > 0.1:
        print("🚨 **100% 确定：多目标代码内部引入了额外计算/梯度噪声/逻辑干扰！**")
        print("   目标选择没有问题，全程都是主目标，但效果变差了！")
        print("   问题出在：token_gradients_mul 内部逻辑！")

    print("="*80)

# ===================== 运行 =====================
if __name__ == "__main__":
    ppl_suffix = "ppl" if args.use_ppl_filter else ""
    log_dir = pathlib.Path(args.output_path) / "log"

    file_single = log_dir / f'_contrast_{args.loss_type}_{ppl_suffix}_{args.str_init}'
    file_multi = log_dir / f'multi_contrast_{args.loss_type}_{ppl_suffix}_{args.str_init}'

    diagnose_multi_target_issue(file_single, file_multi, max_id=200)