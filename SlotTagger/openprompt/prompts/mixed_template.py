import os
import string
from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template

import torch
from torch import nn


class MixedTemplate(Template):
    r"""The Mixed Template class defined by a string of `text`. See more examples in the `tutorial <https://github.com/thunlp/OpenPrompt/blob/ca27491101df0108a8dd753e5b1e79bf591f65d3/tutorial/1.1_mixed_template.py>`_.

    Args:
        model (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        tokenizer (:obj:`PreTrainedTokenizer`): A tokenizer to appoint the vocabulary and the tokenization strategy.
        text (:obj:`Optional[List[str]]`, optional): manual template format. Defaults to None.
    """
    registered_inputflag_names = ["soft_token_ids", "loss_ids", "shortenable_ids"]

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: Optional[str] = None,
                 placeholder_mapping: dict = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, ):

        super().__init__(tokenizer=tokenizer, placeholder_mapping=placeholder_mapping)

        self.raw_embedding = model.get_input_embeddings()
        self.embedding_size = self.raw_embedding.weight.shape[-1]
        self.text = text
        # 因为setter及后面的一系列方法的缘故，这里使用初始文本给self.text赋值会调用许多方法，先后分别是：
        # self._text = text，这里是将初始文本赋值给self._text
        # parsed = self.parse_text(self._text)，这里是解析初始文本
        # self._text = parsed，这里是将处理好的文本赋值给self._text
        # self.prepare()，调用prepare方法

    def get_default_soft_token_ids(self) -> List[int]:
        return self.soft_token_ids

    def prepare(self):
        r"""get the soft token indices ( soft_token_ids ) for the template

        ``"soft_id"`` can be used to reference the previous soft token, which means these tokens use the same embeddings.
        **Note that ``"soft_id"`` should have index start from 1 but not 0**

        e.g. when self.text is ``'{"soft": None} {"soft": "the", "soft_id": 1} {"soft": None} {"soft": "it", "soft_id": 3} {"soft_id": 1} {"soft": "was"} {"mask"}'``,
        output is [1, 2, 3, 4, 2, 5, 0]
        """
        # "soft_id"指定的数字只是用来方便在模板中指示哪几个单词使用相同的embedding，具体的soft_token_id还是会按照出现顺序来排序
        # "soft_id"相同时，对应的soft_token_id也会相同
        # 注意区分模板中的单词，None是不带引号的，表示None，其他单词是字符串，带引号

        num_soft_token = 0
        text = []
        soft_token_ids = []
        idx_mp = {}
        emb_mp = {}
        # id_list 是（若干个）soft_token_id，被这里的代码写成soft_id
        # soft_id 与 ”soft_id“是两个东西，前者此处的代码表示soft_token_id，后者在模板中只用来标识相同的单词是否使用相同的embedding。
        # 这里应该是考虑一个相同的词组被设置成相同soft id，或者说是一个稍微复杂的单词，会被tokenizer分词成subwords，因此用list来表示
        # idx_mp用来映射"soft_id“与对应的soft token id
        # emb_mp用来映射 soft token id 与token id的embedding
        for d in self.text:
            # 此处的text是个list，其元素是字典，字典的某个键值是文本
            if "soft" not in d and "soft_id" not in d:
                text.append(d)
                soft_token_ids.append(0)
                continue

            old_num = num_soft_token

            if "soft_id" in d:
                if not isinstance(d["soft_id"], int) or d["soft_id"] <= 0:
                    raise ValueError(f'soft_id should be integer greater than zero, but get {d["soft_id"]}')
                if d["soft_id"] in idx_mp:
                    # 如果前面已经出现过这个”soft_id“，那么text中只需要给个soft None占位即可
                    id_list = idx_mp[d["soft_id"]]
                    text.extend([{"soft": None} for _ in range(len(id_list))])
                    soft_token_ids.extend(id_list)
                    continue
                else:
                    if "soft" not in d: d["soft"] = None
                    # 这个判断表示的是，如果这个"soft_id”前面没出现过，并且只出现了"soft_id",而没有出现"soft"，那么就给这个soft赋None
                    # 其他情况直接pass

            if d["soft"] is None:
                if "duplicate" in d:
                    if "same" in d and d["same"]:
                        num_soft_token += 1
                        id_list = [num_soft_token for _ in range(len(d["duplicate"]))]
                    else:
                        num_soft_token += d["duplicate"]
                        id_list = list(range(old_num + 1, num_soft_token + 1))
                else:
                    num_soft_token += 1
                    id_list = [num_soft_token]
                text.extend([{"soft": ""} for _ in range(len(id_list))])
            else:
                token_ids = self.tokenizer(d["add_prefix_space"] + d["soft"], add_special_tokens=False)["input_ids"]
                surface_forms = self.tokenizer.convert_ids_to_tokens(token_ids)
                # 一个soft类型的单词，可能被分词为好几个subwords，因此这里用id list，
                assert len(token_ids) == len(surface_forms)
                num_soft_token += len(token_ids)
                id_list = list(range(old_num + 1, num_soft_token + 1))
                for idx, soft_id in enumerate(id_list):
                    emb_mp[soft_id] = token_ids[idx]

                text.extend([{"soft": surface_form} for surface_form in surface_forms])
            soft_token_ids.extend(id_list)

            if "soft_id" in d:
                idx_mp[d["soft_id"]] = id_list

        self.num_soft_token = num_soft_token
        self.text = text
        self.soft_token_ids = soft_token_ids

        # Generate the embedding needed for soft tokens

        self.soft_embedding = nn.Embedding(1 + self.num_soft_token, self.embedding_size)
        # nn.Embedding(num_embs,emb_dim)的意思是创建一个词嵌入模型，num_embs代表词表大小, embed_dim代表你嵌入维度
        for soft_id, token_id in emb_mp.items():
            self.soft_embedding.weight.data[soft_id, :] = self.raw_embedding.weight.data[token_id, :].clone().detach().requires_grad_(True)

        # if "post_processing" in d:  #     if d["post_processing"] == "mlp":  #         pass # TODO one mlp or more than one  #     else:  #         raise ValueError(f'post_processing of {d["post_processing"]} is not supported yet')

    def parse_text(self, text: str) -> List[Dict]:
        parsed = []
        i = 0
        while i < len(text):
            d = {"add_prefix_space": ' ' if (i > 0 and text[i - 1] == ' ') else ''}
            while i < len(text) and text[i] == ' ':
                d["add_prefix_space"] = ' '
                i = i + 1
            if i == len(text): break

            if text[i] != self.mixed_token_start:
                j = i + 1
                while j < len(text):
                    if text[j] == self.mixed_token_start:
                        break
                    j = j + 1
                d["text"] = text[i:j].rstrip(' ')
                i = j

            else:
                j = i + 1
                mixed_token_cnt = 1  # { {} {} } nested support
                while j < len(text):
                    if text[j] == self.mixed_token_end:
                        mixed_token_cnt -= 1
                        if mixed_token_cnt == 0: break
                    elif text[j] == self.mixed_token_start:
                        mixed_token_cnt += 1
                    j = j + 1
                if j == len(text):
                    raise ValueError(
                        f"mixed_token_start {self.mixed_token_start} at position {i} has no corresponding mixed_token_end {self.mixed_token_end}")
                dict_str = '{' + text[i + 1:j] + '}'
                try:
                    val = eval(dict_str)
                    if isinstance(val, set):
                        val = {k: None for k in val}
                    d.update(val)
                except:
                    import traceback
                    print(traceback.format_exc())
                    print(f"syntax error in {dict_str}")
                    exit()
                i = j + 1

            parsed.append(d)

        return parsed

    def on_text_set(self):
        """
        when template text was set

        1. parse text

        2. generate parameter needed
        """

        self.text = self.parse_text(self.text)
        self.prepare()

    def incorporate_text_example(self, example: InputExample):
        text = self.text.copy()
        for i, d in enumerate(text):
            if 'placeholder' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x: x)(getattr(example, d['placeholder']))
            elif 'meta' in d:
                text[i] = d["add_prefix_space"] + d.get("post_processing", lambda x: x)(example.meta[d['meta']])
            elif 'soft' in d:
                text[i] = d['soft'];  # unused
            elif 'mask' in d:
                text[i] = '<mask>'
            elif 'special' in d:
                text[i] = d['special']
            elif 'text' in d:
                text[i] = d["add_prefix_space"] + d['text']
            else:
                raise ValueError(f'can not parse {d}')
        return text

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        raw_embeds = self.raw_embedding(batch['input_ids'])
        soft_embeds = self.soft_embedding(batch['soft_token_ids'])
        inputs_embeds = torch.where((batch['soft_token_ids'] > 0).unsqueeze(-1), soft_embeds, raw_embeds)

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        return batch
