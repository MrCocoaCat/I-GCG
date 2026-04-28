import torch
import pandas as pd
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===================== 【基础配置】 =====================
MODEL_PATH = r"D:\Model\Llama-2-7b-chat-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GEN_PER_POS = 10    # 每个位置生成的句子数量
TOP_K = 20         # 最终取最优的数量
TEMPERATURE = 0.7  # 采样温度

# CSV 路径
CSV_PATH = r"D:\github\llm-attacks\data\advbench\harmful_behaviors.csv"
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_file = f"output_{timestamp}.json"

# ===================== 【加载模型&分词器】 =====================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

# Llama 必须设置
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model.eval()

# ===================== 全局变量 =====================
BASE_SENTENCE = ""
base_ids = None
base_len = 0
ALL_POSITIONS = []
base_embedding = None

# ===================== 【核心函数1：单位置生成】 =====================
def generate_by_pos(original_ids, mask_pos, num=1):
    generated = []
    for _ in range(num):
        new_ids = original_ids.clone()

        with torch.no_grad():
            logits = model(new_ids).logits

        # 对指定位置做采样
        logits_at_pos = logits[0, mask_pos] / TEMPERATURE
        probs = torch.softmax(logits_at_pos, dim=-1)
        new_token = torch.multinomial(probs, 1)
        new_ids[:, mask_pos] = new_token

        # 解码
        sent = tokenizer.decode(new_ids[0], skip_special_tokens=True).strip()

        # 过滤无效句子
        if sent and sent != BASE_SENTENCE and sent not in generated:
            generated.append(sent)

    return generated

# ===================== 【核心函数2：快速计算相似度】 =====================
def get_similarity_fast(sentence):
    ids = tokenizer.encode(
        sentence,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=True,
        max_length=base_len + 2
    ).to(DEVICE)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
        emb = out.hidden_states[-1].mean(dim=1).squeeze()

    return torch.cosine_similarity(base_embedding, emb, dim=0).item()

# ===================== 【你的原版主逻辑 —— 完全不动】 =====================
def run_original_code(sentence):
    global BASE_SENTENCE, base_ids, base_len, ALL_POSITIONS
    global base_embedding

    BASE_SENTENCE = sentence

    print("=" * 100)
    print(f"🔥 逐位置掩码 | 每个位置生成{GEN_PER_POS} | 嵌入排序TOP{TOP_K} | 全位置覆盖")
    print("=" * 100)

    base_ids = tokenizer.encode(
        BASE_SENTENCE,
        return_tensors="pt",
        add_special_tokens=False
    ).to(DEVICE)

    base_len = base_ids.shape[1]
    ALL_POSITIONS = list(range(base_len))

    print(f"📝 原始句子: {BASE_SENTENCE}")
    print(f"📏 Token长度: {base_len}")
    print(f"📍 全部掩码位置: {ALL_POSITIONS}\n")

    # ===================== 【预计算原句嵌入】 =====================
    with torch.no_grad():
        base_out = model(base_ids, output_hidden_states=True)
        base_embedding = base_out.hidden_states[-1].mean(dim=1).squeeze()

    # ===================== 【步骤1：逐位置生成】 =====================
    print(f"🔁 开始生成 | 每个位置生成{GEN_PER_POS}个句子：\n")
    all_candidates = []

    for pos in ALL_POSITIONS:
        print(f"📍 处理位置 {pos}")
        sents = generate_by_pos(base_ids, pos, num=GEN_PER_POS)

        for sent in sents:
            all_candidates.append(sent)
            print(f"   ✅ {sent}")
        print()

    # ===================== 【步骤2：去重 + 排序】 =====================
    print("🔍 计算嵌入相似度并排序...\n")
    all_candidates = list(set(all_candidates))  # 去重

    # 打分
    scored = [(sent, get_similarity_fast(sent)) for sent in all_candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_results = scored[:TOP_K]

    # ===================== 【步骤3：输出最终结果】 =====================
    print("=" * 100)
    print(f"🎯 最终最优 TOP{TOP_K} 个句子")
    print("=" * 100)

    for i, (sent, score) in enumerate(top_results):
        print(f"{i+1:02d}. | 相似度:{score:.4f} | {sent}")

    print("=" * 100)
    print(f"✅ 完成 | 总候选：{len(all_candidates)} | 展示TOP{TOP_K}")
    print("=" * 100)

    return top_results

# ===================== 批量 CSV 处理 =====================
if __name__ == '__main__':
    # 初始化输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([], f, indent=4, ensure_ascii=False)

    df = pd.read_csv(CSV_PATH)
    total = len(df)

    print(f"✅ 开始批量处理 {total} 条数据\n")

    for idx, (raw_id, row) in enumerate(df.iterrows(), 1):
        print(f"\n**************************** 处理第 {idx}/{total} 条 ****************************\n")
        goal = str(row["goal"]).strip()
        target = str(row["target"]).strip()

        # 运行你原版代码（所有打印完全不变）
        top_results = run_original_code(target)

        # 组装结果
        result_item = {
            "id": int(raw_id) + 1,
            "behaviour": goal,
            "target": target,
            "target_similar": [
                {"text": s, "cosine_similarity": round(sc, 6)}
                for s, sc in top_results
            ],
        }

        # 保存
        with open(output_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.append(result_item)
            f.seek(0)
            json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n🎉 全部完成！文件：{output_file}")