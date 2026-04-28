import random
import pandas as pd
import json
import time
import torch
from transformers import AutoTokenizer, AutoModel, pipeline
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 【核心配置】 =====================
NUM_CANDIDATES = 40        # 生成候选改写数量
TOP_K = 30                # 保留相似度最高前30条
MAX_TRIES = 100           # 防无限循环
CSV_PATH = r"D:\github\llm-attacks\data\advbench\harmful_behaviors.csv"
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# 轻量快速句子嵌入模型（计算余弦相似度）
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 标准回译模型（学术首选：英语 ↔ 法语）
TRANSLATION_MODEL_EN_FR = "Helsinki-NLP/opus-mt-en-fr"
TRANSLATION_MODEL_FR_EN = "Helsinki-NLP/opus-mt-fr-en"

# ===================== 【加载所有模型】 =====================
print("\n[1/3] 加载句子向量模型（计算相似度）...")
emb_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)
emb_model = AutoModel.from_pretrained(EMBED_MODEL)
device = "cuda" if torch.cuda.is_available() else "cpu"
emb_model = emb_model.to(device)
emb_model.eval()

print("\n[2/3] 加载英语 → 法语翻译模型...")
trans_en2fr = pipeline(
    "translation",
    model=TRANSLATION_MODEL_EN_FR,
    device=0 if torch.cuda.is_available() else -1
)

print("\n[3/3] 加载法语 → 英语翻译模型...")
trans_fr2en = pipeline(
    "translation",
    model=TRANSLATION_MODEL_FR_EN,
    device=0 if torch.cuda.is_available() else -1
)

print("\n✅ 所有模型加载完成！\n")

# ===================== 【核心：回译改写】 =====================
def back_translate(text):
    try:
        # 英文 → 法文
        fr_text = trans_en2fr(text, max_length=512, truncation=True)[0]["translation_text"]
        # 法文 → 英文
        en_text = trans_fr2en(fr_text, max_length=512, truncation=True)[0]["translation_text"]
        return en_text.strip()
    except Exception as e:
        return text  # 出错则返回原句

# ===================== 【余弦相似度计算】 =====================
def get_embedding(text):
    inputs = emb_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = emb_model(**inputs)

    # 取句向量
    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return emb

def compute_cosine(target, aug_text):
    emb1 = get_embedding(target)
    emb2 = get_embedding(aug_text)
    sim = cosine_similarity(emb1, emb2)[0][0]
    return round(float(sim), 6)

# ===================== 【生成候选（去重 + 防卡死）】 =====================
def generate_rewrites(origin):
    res = {origin}
    tries = 0
    while len(res) < NUM_CANDIDATES and tries < MAX_TRIES:
        rewritten = back_translate(origin)
        res.add(rewritten)
        tries += 1
    return list(res)

# ===================== 【处理 + 保存（格式完全一致）】 =====================
def process_and_save():
    out_file = f"backtranslation_{TIMESTAMP}.json"
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    data = []

    print("=" * 90)
    print("📌 方法：Back-Translation 回译改写 (ACL 2016)")
    print(f"📌 输出文件：{out_file}")
    print(f"📌 总数据量：{total} 条")
    print("=" * 90)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        goal = str(row["goal"]).strip()
        target = str(row["target"]).strip()

        print(f"[{idx}/{total}] 处理 → {target[:70]}...")

        # 1. 生成改写
        rewrites = generate_rewrites(target)

        # 2. 计算相似度
        sim_list = []
        for text in rewrites:
            score = compute_cosine(target, text)
            sim_list.append({
                "text": text,
                "cosine_similarity": score
            })

        # 3. 按相似度从高到低排序
        sim_list.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        # 4. 取 TOP_K
        topk_result = sim_list[:TOP_K]

        # 5. 组装你要的格式
        data.append({
            "id": idx,
            "behaviour": goal,
            "target": target,
            "target_similar": topk_result
        })

    # 保存JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ 处理完成！文件已保存 → {out_file}\n")

# ===================== 【运行】 =====================
if __name__ == "__main__":
    random.seed(42)
    process_and_save()
    print("🎉 全部任务完成！")