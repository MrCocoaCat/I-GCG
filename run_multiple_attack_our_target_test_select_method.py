import threading
import time
import random
import datetime
from run_single_attack_base import (run_single_process,run_single_process_select_method)
import os
# make the timestamp utc-8
import argparse
# ===================== 1. 命令行参数解析 =====================
# 创建参数解析器对象，用于处理外部传入的参数
parser = argparse.ArgumentParser()
parser.add_argument('--defense', type=str, default="no_defense")
#parser.add_argument('--behaviors_config', type=str, default="behaviors_config.json")
parser.add_argument('--output_path', type=str,default='ours')

# ===================== 2. 全局配置初始化 =====================

args = parser.parse_args()
device_list = [0]
# 提取防御策略参数（从命令行传入或默认值）
defense=args.defense
timestamp = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
# 拼接最终输出路径：根目录(Our_GCG_target_len_20) + 用户指定路径 + 时间戳
output_path=os.path.join("test_select_method",args.output_path)
output_path=os.path.join(output_path,str(timestamp))

behaviors_config="adv_similar.json"
# 生成攻击行为ID列表：1-50（对应配置文件中50个有害行为）
behavior_id_list = [i + 1 for i in range(50)]
# behavior_id_list = list(range(50, 0, -1))
# add id to black_list to skip the id
# 以下为预留的黑白名单配置（注释状态，可启用）：
# 黑名单：添加需要跳过的攻击ID，执行时会过滤掉这些ID
# black_list = []
# 白名单：仅执行列表内的攻击ID，其余跳过
# white_list =[]

# 启用黑名单过滤（需取消注释）：从行为列表中移除黑名单ID
# behavior_id_list = [i for i in behavior_id_list if i not in black_list]
# 启用白名单过滤（需取消注释）：仅保留白名单内的ID
# behavior_id_list = [i for i in behavior_id_list if i in white_list]

# ===================== 3. GPU资源管理类 =====================
# 定义Card类：封装单个GPU卡的ID和独占锁
class Card:
    def __init__(self, id):
        self.id = id
        self.lock = threading.Lock()

# 定义ResourceManager类：管理所有GPU资源的申请/释放
class ResourceManager:
    def __init__(self, device_list):
        self.cards = [Card(i) for i in device_list]

    def request_card(self):
        for card in self.cards:
            if card.lock.acquire(False):
                return card
        return None

    def release_card(self, card):
        card.lock.release()

# ===================== 4. 工作线程任务逻辑 =====================
# 定义worker_task函数：每个线程执行的核心逻辑，循环处理攻击任务
def worker_task(task_list, resource_manager):
    while True:
        task = None
        with task_list_lock:
            if task_list:
                task = task_list.pop()

        if task is None:  # No more tasks left
            break

        card = resource_manager.request_card()
        while card is None:  # Keep trying until a card becomes available
            time.sleep(0.01)
            card = resource_manager.request_card()

        print(f"Processing task {task} using card {card.id}")
        # 实际执行的逻辑，其他的代码都是并行处理逻辑
        num_steps = 2
        batch_size = 2
        model_path= r"D:\Model\Llama-2-7b-chat-hf"
        # model_path = "/home/liyubo/Model/Llama-2-7b-chat-hf"

        run_single_process_select_method(behavior_id = task, device = card.id, output_path = output_path,
                                         defense = defense,behaviors_config = behaviors_config, num_steps = num_steps,
                                         batch_size=batch_size,loss_type="cross_entropy",
                                         model_path=model_path)

        run_single_process_select_method(behavior_id=task, device=card.id, output_path=output_path,
                                         defense=defense, behaviors_config=behaviors_config, num_steps=num_steps,
                                         batch_size=batch_size, loss_type="cross_entropy",
                                         model_path=model_path,
                                         use_weighted_sample = "True")


        resource_manager.release_card(card)

# ===================== 5. 主程序入口 =====================
# 将攻击行为ID列表赋值给tasks（待处理任务列表）
tasks = behavior_id_list
# 创建任务列表的全局锁：保证多线程对task_list的修改是线程安全的
task_list_lock = threading.Lock()
# 初始化GPU资源管理器：传入可用的GPU列表
resource_manager = ResourceManager(device_list)

# Create and start 8 worker threads
# 创建工作线程：线程数 = GPU数量（避免线程数过多导致资源竞争）
threads = [threading.Thread(target=worker_task, args=(tasks, resource_manager)) for _ in range(len(device_list))]

for t in threads:
    t.start()

for t in threads:
    t.join()

print("All tasks completed!")