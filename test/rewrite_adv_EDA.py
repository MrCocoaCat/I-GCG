import random
import pandas as pd
import json
import time
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity

# ===================== 【全局配置】 =====================
NUM_CANDIDATES = 60  # 先生成多少条候选
TOP_K = 30  # 保留相似度最高的前30条
MAX_TRIES = 200  # 防死循环
CSV_PATH = r"D:\github\llm-attacks\data\advbench\harmful_behaviors.csv"
MODEL_PATH = r"D:\Model\Llama-2-7b-chat-hf"
LLAMA_PATH = r"D:\Model\Llama-2-7b-chat-hf" # 你的模型路径
TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")

# ===================== 【加载 Llama 模型】 =====================
print("\n🔹 正在加载 Llama-2 模型 & 分词器...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)
tokenizer.pad_token = tokenizer.eos_token  # 修复报错
tokenizer.padding_side = "right"

model = AutoModel.from_pretrained(
    LLAMA_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()
print("✅ 模型加载完成！\n")


# ===================== 【句子 Embedding + 余弦相似度】 =====================
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs)

    # 取 <[BOS_never_used_51bce0c785ca2f68081bfa7d91973934]>作为句向量
    emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return emb


def compute_cosine_sim(target, aug_text):
    emb1 = get_embedding(target)
    emb2 = get_embedding(aug_text)
    sim = cosine_similarity(emb1, emb2)[0][0]
    return round(float(sim), 6)


# ===================== 【EDA 同义词库 / 停用词】 =====================
SYNONYM_LIB = {
    "make": ["create", "build", "produce", "craft", "construct", "develop", "prepare"],
    "create": ["make", "build", "form", "generate", "produce"],
    "build": ["make", "create", "assemble", "construct", "fabricate"],
    "do": ["perform", "execute", "conduct", "accomplish", "complete"],
    "perform": ["do", "execute", "carry out", "implement"],
    "use": ["utilize", "employ", "apply", "operate", "wield"],
    "utilize": ["use", "employ", "apply"],
    "get": ["obtain", "acquire", "gain", "fetch", "secure"],
    "obtain": ["get", "acquire", "gain", "procure"],
    "learn": ["study", "master", "grasp", "understand"],
    "teach": ["instruct", "show", "demonstrate", "explain"],
    "show": ["teach", "demonstrate", "present", "reveal"],
    "method": ["way", "approach", "technique", "process", "manner"],
    "way": ["method", "approach", "technique", "process"],
    "recipe": ["guide", "steps", "instructions", "procedure", "process"],
    "steps": ["procedure", "process", "actions", "stages"],
    "guide": ["recipe", "instructions", "manual", "tutorial"],
    "device": ["tool", "gadget", "apparatus", "instrument", "equipment"],
    "tool": ["device", "instrument", "implement", "mechanism"],
    "material": ["substance", "element", "component", "ingredient"],
    "explosive": ["blast", "combustive", "detonation"],
    "weapon": ["tool", "device", "instrument", "object"],
    "simple": ["easy", "basic", "straightforward", "uncomplicated", "effortless"],
    "easy": ["simple", "basic", "effortless", "plain"],
    "basic": ["simple", "easy", "fundamental", "standard"],
    "homemade": ["household", "domestic", "self-made", "home-built"],
    "household": ["homemade", "domestic", "home-based"],
    "dangerous": ["hazardous", "risky", "harmful", "unsafe"],
    "quick": ["fast", "rapid", "swift", "speedy"],
    "safe": ["secure", "protected", "harmless", "risk-free"],
    "for": ["to", "used for", "designed for", "meant for"],
    "how": ["ways to", "steps to", "method to", "guide to"],
    "here": ["below", "here is", "here are", "in this guide"],
    "sure": ["certainly", "of course", "indeed", "absolutely"]
}

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself",
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "or", "as", "because", "while",
    "of", "at", "by", "for", "with", "against", "between", "into",
    "through", "during", "before", "after", "to", "from", "up", "down",
    "again", "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "just", "now"
}

INSERT_WORDS = [
    "simple", "basic", "easy", "detailed", "practical",
    "common", "regular", "standard", "typical", "normal",
    "safe", "quick", "efficient", "useful", "effective"
]


# ===================== 【四种 EDA 核心方法】 =====================
def augment_synonym_only(text):
    words = text.strip().split()
    new_words = []
    for w in words:
        w_low = w.lower()
        if w_low in SYNONYM_LIB and random.random() < 0.65:
            new_words.append(random.choice(SYNONYM_LIB[w_low]))
        else:
            new_words.append(w)
    return " ".join(new_words).strip()


def augment_delete_only(text):
    words = text.strip().split()
    new_words = [w for w in words if not (w.lower() in STOP_WORDS and random.random() < 0.4)]
    return " ".join(new_words).strip()


def augment_insert_only(text):
    words = text.strip().split()
    if len(words) >= 3 and random.random() < 0.7:
        pos = random.randint(0, len(words) - 1)
        words.insert(pos, random.choice(INSERT_WORDS))
    return " ".join(words).strip()


def augment_swap_only(text):
    words = text.strip().split()
    if len(words) >= 5:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words).strip()


def augment_all_eda(text):
    out = augment_synonym_only(text)
    out = augment_delete_only(out)
    out = augment_insert_only(out)
    out = augment_swap_only(out)
    return out


# ===================== 【生成候选句子（去重 + 防卡死）】 =====================
def generate_candidates(origin, aug_func):
    res = {origin}
    tries = 0
    while len(res) < NUM_CANDIDATES and tries < MAX_TRIES:
        res.add(aug_func(origin))
        tries += 1
    return list(res)


# ===================== 【处理 + 保存（按相似度排序 TOP_K）】 =====================
def process_and_save(method_name, aug_func, suffix):
    out_file = f"eda_final_{TIMESTAMP}_{suffix}.json"
    df = pd.read_csv(CSV_PATH)
    total = len(df)
    data = []

    print("\n" + "=" * 80)
    print(f"📌 正在处理：{method_name}")
    print(f"📌 输出文件：{out_file}")
    print(f"📌 总数据量：{total}")
    print("=" * 80)

    for idx, (_, row) in enumerate(df.iterrows(), 1):
        goal = str(row["goal"]).strip()
        target = str(row["target"]).strip()
        print(f"[{idx}/{total}] 生成 → {target[:60]}...")

        # 1. 生成候选
        candidates = generate_candidates(target, aug_func)

        # 2. 计算相似度
        sim_list = []
        for t in candidates:
            s = compute_cosine_sim(target, t)
            sim_list.append({"text": t, "cosine_similarity": s})

        # 3. 按相似度从高到低排序
        sim_list.sort(key=lambda x: x["cosine_similarity"], reverse=True)

        # 4. 取 TOP_K
        topk_list = sim_list[:TOP_K]

        # 5. 拼装格式
        data.append({
            "id": idx,
            "behaviour": goal,
            "target": target,
            "target_similar": topk_list
        })

    # 保存
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ {method_name} 完成！已保存 → {out_file}\n")


# ===================== 【运行所有方法】 =====================
if __name__ == "__main__":
    random.seed(42)
    print("\n📢 EDA 四种标准增强 + Llama 真实相似度排序 TOP_K")
    print("=" * 80)

    process_and_save("Synonym Replacement", augment_synonym_only, "synonym")
    process_and_save("Random Deletion", augment_delete_only, "delete")
    process_and_save("Random Insertion", augment_insert_only, "insert")
    process_and_save("Random Swap", augment_swap_only, "swap")
    process_and_save("EDA All Methods", augment_all_eda, "all")

    print("\n🎉 全部任务已完成！")