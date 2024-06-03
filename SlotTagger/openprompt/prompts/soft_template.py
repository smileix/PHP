import os

from torch.nn.parameter import Parameter
from openprompt.utils.logging import logger

from openprompt.data_utils import InputExample, InputFeatures
from typing import *

from transformers import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from openprompt import Template
from openprompt.prompts import ManualTemplate, ManualVerbalizer

import torch
from torch import nn


class SoftTemplate(Template):
    r"""This is the implementation of `The Power of Scale for Parameter-Efficient
    Prompt Tuning <https://arxiv.org/pdf/2104.08691v1.pdf>`_ . Similar to :obj:`PrefixTuningTemplate`,
    This template also does not need any textual template. Addition tokens are directly
    concatenated into the input ids. There are two initializations of the new tokens.
    (1). random initialization. (2) initialize with the tokens of the plm (We simply take
    the first n_tokens similar to their implementation).
    """
    registered_inputflag_names = ["loss_ids", "shortenable_ids"]

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, text: Optional[str] = None,
                 soft_embeds: Optional[torch.FloatTensor] = None, num_tokens: int = 20, initialize_from_vocab: Optional[bool] = True,
                 random_range: Optional[float] = 0.5, placeholder_mapping: dict = {'<text_a>': 'text_a', '<text_b>': 'text_b'}, ):
        super().__init__(tokenizer=tokenizer, placeholder_mapping=placeholder_mapping)
        self.raw_embedding = model.get_input_embeddings()
        self.raw_embedding.requires_grad_(False)
        self.model_is_encoder_decoder = model.config.is_encoder_decoder
        self.random_range = random_range
        self.num_tokens = num_tokens
        self.initialize_from_vocab = initialize_from_vocab

        self.text = text
        # self.default_text1 = {"placeholder<text_a> <mask>"
        # self.default_text2 = "<text_a> <text_b> <mask>".split()

        if soft_embeds is not None:
            self.soft_embeds = soft_embeds
            self.num_tokens = len(soft_embeds)
        else:
            if self.num_tokens > 0:
                self.generate_parameters()

    def on_text_set(self):
        self.text = self.parse_text(self.text)

    def wrap_one_example(self, example) -> List[Dict]:  # TODO this automatic generated template may not be able to process diverse data format.
        if self.text is None:
            logger.warning("You didn't provide text template for softprompt. Using default template, is this intended?")
            if example.text_b is None:
                self.text = self.default_text1
            else:
                self.text = self.default_text2
        return super().wrap_one_example(example)

    def generate_parameters(self) -> None:
        """
        generate parameters needed for soft tokens embedding in soft-prompt
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        # 默认是从词表初始化，整个soft template的核心思路，就是在初始文本嵌入模板之后，在句首加入一段20token长度的soft embedding，
        # 然后与raw embedding横着拼一块，训练时，soft embedding的前20个token处的embedding可以变化，只不过这种变化并不依赖输入。
        # 因为此处定义的soft embedding只是nn parameter，而不是nn embedding，没有ff方法，
        # 更像是bias，而不是weight
        # 在之前的实验中，用soft不多，但我记得用soft template效果好像还没manual template的高
        if self.initialize_from_vocab:
            soft_embeds = self.raw_embedding.weight[:self.num_tokens].clone().detach()
        else:
            soft_embeds = torch.FloatTensor(self.num_tokens, self.raw_embedding.weight.size(1)).uniform_(-self.random_range, self.random_range)
        # 参数在这里设置，这里定义的soft embeds，只用了nn的parameter方法，有可训练的参数，但没forward函数，没办法调用，
        # 因此这里采取的是new_embeds = raw_embeds(x) + soft_embeds_repeated的方式得到处理过后的embedding
        self.soft_embeds = nn.Parameter(soft_embeds, requires_grad=True)

    def process_batch(self, batch: Union[Dict, InputFeatures]) -> Union[Dict, InputFeatures]:
        """
        Convert input_ids to inputs_embeds
        for normal tokens, use the embedding layer of PLM
        for soft tokens, use a new embedding layer which is initialized with their corresponding embedding of hard tokens
        """
        inputs_embeds = self.raw_embedding(batch['input_ids'])
        batch_size = inputs_embeds.size(0)
        # 参数在这里设置，这里定义的soft embeds，只用了nn的parameter方法，有可训练的参数，但没forward函数，没办法调用，
        # 因此这里采取的是new_embeds = raw_embeds(x) + soft_embeds_repeated的方式得到处理过后的embedding
        if self.num_tokens > 0:
            soft_embeds = self.soft_embeds.repeat(batch_size, 1, 1)
            # repeat()表示复制，各参数表示对应维度复制倍数，n维tensor复制时最多指定n+1个参数，会自动进行unsqueeze(0)处理扩充维度再repeat
            inputs_embeds = torch.cat([soft_embeds, inputs_embeds],
                                      1)  # 复制之后，这里的soft_embeds的shape是(bsz, num_tokens, emb_dim)  # 这里的input_embeds的shape是(bsz, max_len, emb_dim),拼接之后，变成(bszm, num_tokens + max_len, emb_dim)  # 不管初始是什么任务，提示范式都将其重构成了单句分类任务，这里的处理相当于句子长度变长了，在每个句子的句首拼接了一个soft embedding，  # 这并不影响单句分类，因为label都是以句子为单位的，而不是序列标注问题的以token为单位。

        batch['input_ids'] = None
        batch['inputs_embeds'] = inputs_embeds
        if 'attention_mask' in batch and self.num_tokens > 0:
            am = batch['attention_mask']
            batch['attention_mask'] = torch.cat([torch.ones((batch_size, self.num_tokens), dtype=am.dtype, device=am.device), am], dim=-1)
        return batch

    def post_processing_outputs(self, outputs: torch.Tensor):
        r"""Post processing the outputs of language models according
        to the need of template. Most templates don't need post processing,
        The template like SoftTemplate, which appends soft template as a module
        (rather than a sequence of input tokens) to the input,
        should remove the outputs on these positions to keep the seq_len the same
        """
        if not self.model_is_encoder_decoder:
            outputs.logits = outputs.logits[:, self.num_tokens:, :]
        return outputs
