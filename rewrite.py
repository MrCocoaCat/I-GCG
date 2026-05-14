import random
import pandas as pd
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 【核心配置】 =====================
NUM_PARAPHRASES = 10
CSV_PATH = r"D:\github\llm-attacks\data\advbench\harmful_behaviors.csv"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
LLAMA_MODEL = r"D:\Model\Llama-2-7b-chat-hf"

# ===================== 加载Llama模型 =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("\n[加载 Llama2 模型中...]")

tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    LLAMA_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)

random.seed(42)
torch.manual_seed(42)

# ===================== 通用生成函数（收敛不瞎编） =====================
def llama_generate(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.2,      # 极低随机，不瞎编
            top_p=0.85,
            do_sample=False,      # 关闭采样，更稳定
            pad_token_id=tokenizer.eos_token_id
        )
    raw_out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()
    # 清洗：过滤带 French:/English: 垃圾行、空行、太短行
    lines = []
    for line in raw_out.split("\n"):
        s = line.strip()
        if not s:
            continue
        # 剔除带标签的垃圾行
        if s.lower().startswith("french:") or s.lower().startswith("english:"):
            continue
        if len(s) > 15:
            lines.append(s)
    return lines

# ===================== 1. 英文 → 标准一句法语 =====================
def en_to_fr_single(en_text):
    prompt = f"""Translate this English sentence to proper French.
Only give one French sentence, no extra explanation, no extra text.
English: {en_text}"""
    res = llama_generate(prompt)
    return res[0] if res else en_text

# ===================== 2. 一句法语 → 生成10句不同同义英文（强约束） =====================
def fr_to_en_10_variants(fr_text):
    prompt = f"""Generate 10 different English sentences with the exact same meaning as this French sentence.
Do not write French, do not write explanations, do not add labels, only output 10 natural English lines.
French sentence: {fr_text}"""
    res = llama_generate(prompt)
    return res[:NUM_PARAPHRASES]

# ===================== 完整流程 + 日志打印 =====================
def back_translate_with_log(origin_en):
    print("\n" + "="*70)
    print(f"原始英文: {origin_en}")
    print("="*70)

    # Step1 英译法
    fr_single = en_to_fr_single(origin_en)
    print(f"【1. 对应法语翻译】: {fr_single}")

    # Step2 法语生成10条不同英文
    en_variant_list = fr_to_en_10_variants(fr_single)
    print(f"\n【2. 同一法语生成10种不同英文表达】")
    for idx, txt in enumerate(en_variant_list, 1):
        print(f"   En{idx}: {txt}")

    # 去重 + 兜底补满10条
    en_variant_list = list(dict.fromkeys(en_variant_list))
    while len(en_variant_list) < NUM_PARAPHRASES:
        en_variant_list.append(origin_en)
    en_variant_list = en_variant_list[:NUM_PARAPHRASES]

    print(f"\n✅ 本轮10条英文同义句生成完毕\n")
    return en_variant_list

# ===================== 批量处理保存JSON =====================
def process_and_save():
    out_file = f"llama_fr2en_clean_{TIMESTAMP}.json"
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    data = []

    print("=" * 80)
    print("📌 流程：英文→法语 → 生成10条同义英文翻译（强约束防跑偏）")
    print("=" * 80)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        goal = str(row["goal"]).strip()
        target = str(row["target"]).strip()
        print(f"\n========== 处理样本 [{idx}/{total}] ==========")
        para_list = back_translate_with_log(target)
        sim_item_list = [{"text": text} for text in para_list]
        data.append({
            "id": idx,
            "behaviour": goal,
            "target": target,
            "target_similar": sim_item_list
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 全部处理完成！已保存 → {out_file}")

if __name__ == "__main__":
    process_and_save()