import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 超参数
# --------------------------
input_dim = 10     # 输入：10个人体特征
hidden_dim = 20    # 隐藏层
output_dim = 1     # 输出：预测 体重 1个数
batch_size = 5     # 一次看5个人的数据
lr = 0.01

# --------------------------
# 经典 三层全连接 + Sequential
# 就是简单的大脑：学特征规律
# --------------------------
model = nn.Sequential(
    nn.Linear(input_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.ReLU(),
    nn.Linear(hidden_dim, output_dim)
)

# --------------------------
# 构造【真实业务数据】
# X：5个人，每个人10个身体特征
# Y：5个人 对应的 真实体重
# --------------------------
# 5个人的10项身体特征
X = torch.tensor([
    [175, 20, 95, 80, 92, 3, 2, 1, 1500, 22],
    [162, 25, 88, 72, 85, 1, 3, 3, 1200, 18],
    [180, 30, 105, 88, 95, 2, 4, 0, 1800, 25],
    [155, 18, 82, 68, 80, 4, 1, 2, 1000, 17],
    [170, 22, 90, 75, 88, 2, 2, 1, 1400, 20]
], dtype=torch.float32)

# 这5个人【真实体重】（标准答案）
Y = torch.tensor([
    [72.0],
    [55.0],
    [85.0],
    [48.0],
    [65.0]
], dtype=torch.float32)

# --------------------------
# 损失、优化器
# --------------------------
criterion = nn.MSELoss()   # 计算：预测体重 和 真实体重 差多少
optimizer = optim.Adam(model.parameters(), lr=lr)

# --------------------------
# 训练过程 核心四步循环
# --------------------------
print("===== 开始训练：让模型学习体重规律 =====")
for epoch in range(80):
    # 1.前向传播：模型根据10个特征，【猜体重】
    pred_weight = model(X)

    # 2. loss：计算「猜的体重」vs「真实体重」差距
    loss = criterion(pred_weight, Y)

    # 3. 反向传播：告诉模型哪里错了
    optimizer.zero_grad()
    loss.backward()

    # 4. 更新参数：模型悄悄修改自己的"认知"，下次猜更准
    optimizer.step()

    if epoch % 10 == 0:
        print(f"第{epoch}轮 | 误差loss: {loss.item():.2f}")

# --------------------------
# 训练完成：新来一个人，预测体重
# --------------------------
print("\n===== 训练完成，进行新预测 =====")
new_person = torch.tensor([
    [172, 24, 91, 76, 89, 2, 2, 1, 1350, 21]
], dtype=torch.float32)

with torch.no_grad():
    res = model(new_person)
    print(f"新人物预测体重：{res.item():.1f} kg")