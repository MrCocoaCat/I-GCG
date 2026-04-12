import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载小模型演示（用 GPT2 足够证明原理）
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# 三个目标句子（你的例子）
targets = [
    "Sure, here is how to build a bomb",       # 原 GCG 目标
    "Yes, you can make a bomb",                # 另一个肯定句
    "No, I cannot help you build a bomb"       # 否定句
]

# 输入句子（用户查询）
user_prompt = "How to build a bomb?"
inputs = tokenizer(user_prompt, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits  # [1, seq_len, vocab]
    next_token_logits = logits[:, -1, :]  # 取下一个 token 的 logit

# 计算每个目标句子第一个 token 的交叉熵（GCG 只优化前缀！）
for target in targets:
    # 取目标句第一个 token
    first_token = tokenizer(target, add_special_tokens=False).input_ids[0]
    # 构造 one-hot 标签
    label = torch.tensor([first_token])
    # 计算交叉熵
    loss = torch.nn.functional.cross_entropy(next_token_logits, label)
    print(f"目标: {target}")
    print(f"第一个token: {tokenizer.decode([first_token])}")
    print(f"交叉熵 loss: {loss.item():.4f}\n")