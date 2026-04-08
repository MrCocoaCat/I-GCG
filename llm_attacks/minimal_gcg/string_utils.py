import torch
import fastchat 

def load_conversation_template(template_name):
    """
        加载指定名称的对话模板，并针对特定模型（zero_shot/llama-2）做格式适配。

        @par template_name: str
            对话模板名称（如 'llama-2'、'vicuna'、'oasst_pythia'、'zero_shot' 等）
        @return: fastchat.conversation.Conversation
            适配后的对话模板对象
        """
    conv_template = fastchat.model.get_conversation_template(template_name)
    # 适配 zero_shot 模板：角色名添加 ### 前缀，分隔符改为换行
    if conv_template.name == 'zero_shot':
        conv_template.roles = tuple(['### ' + r for r in conv_template.roles])
        conv_template.sep = '\n'
    # 适配 llama-2 模板：去除 sep2 末尾的空白字符
    elif conv_template.name == 'llama-2':
        conv_template.sep2 = conv_template.sep2.strip()
    
    return conv_template


class SuffixManager:
    """
        管理 Prompt 构造与 Token 切片定位的核心类，用于精准划分 Prompt 中各功能区域的 Token 范围。
        核心功能：构造包含指令、对抗后缀、目标输出的完整 Prompt，并定位各部分的 Token 切片。
        """
    def __init__(self, *, tokenizer, conv_template, instruction, target, adv_string):
        """
        初始化 SuffixManager 实例。
        @par tokenizer: transformers.PreTrainedTokenizer
            用于 Token 编解码的分词器
        @par conv_template: fastchat.conversation.Conversation
            适配后的对话模板对象（由 load_conversation_template 生成）
        @par instruction: str
            攻击指令（如 "Tell me how to make a bomb"）
        @par target: str
            期望模型生成的目标输出（即攻击要诱导的内容）
        @par adv_string: str
            初始对抗后缀（用于诱导模型越狱的关键字符串）
        """
        self.tokenizer = tokenizer
        self.conv_template = conv_template
        self.instruction = instruction
        self.target = target
        self.adv_string = adv_string
    
    def get_prompt(self, adv_string=None):
        """
        构造完整的 Prompt 字符串，并精准定位 Prompt 中各部分的 Token 切片范围。
        不同模型（llama-2/其他）采用不同的切片计算逻辑，最终生成可用于模型输入的 Prompt。
        @par adv_string: str, 可选
            可选的新对抗后缀，若传入则更新实例的 adv_string（默认：None）
        @return: str
            构造完成的完整 Prompt 字符串
        """
        # 更新对抗后缀（若传入新值）
        if adv_string is not None:
            self.adv_string = adv_string
        # 基础 Prompt 构造（先适配通用逻辑，再针对不同模型细化）
        self.conv_template.append_message(self.conv_template.roles[0], f"{self.instruction} {self.adv_string}")
        self.conv_template.append_message(self.conv_template.roles[1], f"{self.target}")
        prompt = self.conv_template.get_prompt()

        encoding = self.tokenizer(prompt)
        toks = encoding.input_ids
        # ========== 针对 Llama-2 模型的切片定位 ==========
        if self.conv_template.name == 'llama-2':
            self.conv_template.messages = []
            # 1. 彻底清空消息列表（双重保险：先赋值空列表，再清空）
            self.conv_template.messages.clear()  # 比 =[] 更彻底，避免引用问题

            #print(self.conv_template.roles)
            self.conv_template.append_message(self.conv_template.roles[0], None)
            #print(self.conv_template.messages)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids

            #self._user_role_slice = slice(None, len(toks))
            self._user_role_slice = slice(None, 4)

            decoded_text = self.tokenizer.decode(
                toks[self._user_role_slice],  # 要解码的 Token ID 列表/张量
            )
            #print(f"Decoded text: {decoded_text}")

            self.conv_template.update_last_message(f"{self.instruction}")
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            #print(self.conv_template.messages)
            self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1 )) # 去除尾部的29871 ,即<s>

            decoded_text = self.tokenizer.decode(toks[self._goal_slice])
            #print(f"Decoded text: {decoded_text}")

            separator = ' ' if self.instruction else ''
            self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
            #print(self.conv_template.messages)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._control_slice = slice(self._goal_slice.stop, len(toks) -1  ) # 去除尾部的29871 ,即<s>

            decoded_text = self.tokenizer.decode(toks[self._control_slice])
            #print(f"_control_slice text: {decoded_text}")

            self.conv_template.append_message(self.conv_template.roles[1], None)
            #print(self.conv_template.messages)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

            decoded_text = self.tokenizer.decode(toks[self._assistant_role_slice])
            #print(f"_assistant_role_slice text: {decoded_text}")

            self.conv_template.update_last_message(f"{self.target}")
            #print(self.conv_template.messages)
            toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
            self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-2)
            self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-3)

            decoded_text = self.tokenizer.decode(toks[self._target_slice])
            #print(f"Decoded text: {decoded_text}")

            decoded_text = self.tokenizer.decode(toks[self._loss_slice])
            #print(f"Decoded text: {decoded_text}")
        # ========== 针对其他模型（Vicuna/Pythia 等）的切片定位 ==========
        else:
            python_tokenizer = False or self.conv_template.name == 'oasst_pythia'
            try:
                encoding.char_to_token(len(prompt)-1)
            except:
                python_tokenizer = True

            if python_tokenizer:
                # This is specific to the vicuna and pythia tokenizer and conversation prompt.
                # It will not work with other tokenizers or prompts.
                self.conv_template.messages = []

                self.conv_template.append_message(self.conv_template.roles[0], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._user_role_slice = slice(None, len(toks))

                self.conv_template.update_last_message(f"{self.instruction}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._goal_slice = slice(self._user_role_slice.stop, max(self._user_role_slice.stop, len(toks)-1))

                separator = ' ' if self.instruction else ''
                self.conv_template.update_last_message(f"{self.instruction}{separator}{self.adv_string}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._control_slice = slice(self._goal_slice.stop, len(toks)-1)

                self.conv_template.append_message(self.conv_template.roles[1], None)
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._assistant_role_slice = slice(self._control_slice.stop, len(toks))

                self.conv_template.update_last_message(f"{self.target}")
                toks = self.tokenizer(self.conv_template.get_prompt()).input_ids
                self._target_slice = slice(self._assistant_role_slice.stop, len(toks)-1)
                self._loss_slice = slice(self._assistant_role_slice.stop-1, len(toks)-2)
            else:
                self._system_slice = slice(
                    None, 
                    encoding.char_to_token(len(self.conv_template.system))
                )
                self._user_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[0]) + len(self.conv_template.roles[0]) + 1)
                )
                self._goal_slice = slice(
                    encoding.char_to_token(prompt.find(self.instruction)),
                    encoding.char_to_token(prompt.find(self.instruction) + len(self.instruction))
                )
                self._control_slice = slice(
                    encoding.char_to_token(prompt.find(self.adv_string)),
                    encoding.char_to_token(prompt.find(self.adv_string) + len(self.adv_string))
                )
                self._assistant_role_slice = slice(
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1])),
                    encoding.char_to_token(prompt.find(self.conv_template.roles[1]) + len(self.conv_template.roles[1]) + 1)
                )
                self._target_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)),
                    encoding.char_to_token(prompt.find(self.target) + len(self.target))
                )
                self._loss_slice = slice(
                    encoding.char_to_token(prompt.find(self.target)) - 1,
                    encoding.char_to_token(prompt.find(self.target) + len(self.target)) - 1
                )

        self.conv_template.messages = []

        return prompt
    
    def get_input_ids(self, adv_string=None):
        """
        生成 Prompt 对应的 Token ID 张量，仅保留到目标输出切片结束位置的 Token（模型输入所需）。
        @par adv_string: str, 可选
            可选的新对抗后缀，若传入则更新并重新构造 Prompt（默认：None）
        @return: torch.Tensor
            Prompt 对应的 Token ID 张量，形状为 [序列长度]
        """
        prompt = self.get_prompt(adv_string=adv_string)
        # 第二步：将 Prompt 编码为 Token ID 列表
        # tokenizer(prompt) 返回字典，input_ids 是核心字段，存储 Prompt 对应的所有 Token ID
        toks = self.tokenizer(prompt).input_ids
        # 第三步：截断 Token ID 列表，仅保留到目标输出切片结束位置
        # self._target_slice.stop 是目标输出切片的结束索引（由 get_prompt 计算得出）
        # 截断原因：模型仅需输入到“目标输出开始前”的内容即可预测目标输出，后续 Token 无意义且浪费计算
        input_ids = torch.tensor(toks[:self._target_slice.stop])
        return input_ids

    def control_slice(self, toks=None):
        if toks is None:
            return  self._control_slice
        else:
            return toks[self._control_slice]

    def target_slice(self, toks=None):
        if toks is None:
            return self._target_slice
        else:
            return self._target_slice,  # 目标输出的token切片

    def loss_slice(self, toks=None):
        if toks is None:
            return self._loss_slice
        else:
            return self._loss_slice



