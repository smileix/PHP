import json
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode
from openprompt.data_utils import InputFeatures
import re
from openprompt import Verbalizer
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from openprompt.utils.logging import logger



class ManualVerbalizer(Verbalizer):
    r"""
    The basic manually defined verbalizer class, this class is inherited from the :obj:`Verbalizer` class.

    Args:
        tokenizer (:obj:`PreTrainedTokenizer`): The tokenizer of the current pre-trained model to point out the vocabulary.
        classes (:obj:`List[Any]`): The classes (or labels) of the current task.
        label_words (:obj:`Union[List[str], List[List[str]], Dict[List[str]]]`, optional): The label words that are projected by the labels.
        prefix (:obj:`str`, optional): The prefix string of the verbalizer (used in PLMs like RoBERTa, which is sensitive to prefix space)
        multi_token_handler (:obj:`str`, optional): The handling strategy for multiple tokens produced by the tokenizer.
        post_log_softmax (:obj:`bool`, optional): Whether to apply log softmax post processing on label_logits. Default to True.
    """
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 classes: Optional[List] = None,
                 num_classes: Optional[Sequence[str]] = None,
                 label_words: Optional[Union[Sequence[str], Mapping[str, str]]] = None,
                 prefix: Optional[str] = " ",
                 multi_token_handler: Optional[str] = "first",
                 post_log_softmax: Optional[bool] = True,
                ):
        super().__init__(tokenizer=tokenizer, num_classes=num_classes, classes=classes)
        self.prefix = prefix
        self.multi_token_handler = multi_token_handler
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax

    def on_label_words_set(self):
        super().on_label_words_set()
        self.label_words = self.add_prefix(self.label_words, self.prefix)

         # TODO should Verbalizer base class has label_words property and setter?
         # it don't have label_words init argument or label words from_file option at all

        self.generate_parameters()

    @staticmethod
    def add_prefix(label_words, prefix):
        r"""Add prefix to label words. For example, if a label words is in the middle of a template,
        the prefix should be ``' '``.

        Args:
            label_words (:obj:`Union[Sequence[str], Mapping[str, str]]`, optional): The label words that are projected by the labels.
            prefix (:obj:`str`, optional): The prefix string of the verbalizer.

        Returns:
            :obj:`Sequence[str]`: New label words with prefix.
        """
        new_label_words = []
        if isinstance(label_words[0], str):
            label_words = [[w] for w in label_words]  #wrapped it to a list of list of label words.

        for label_words_per_label in label_words:
            new_label_words_per_label = []
            for word in label_words_per_label:
                if word.startswith("<!>"):
                    new_label_words_per_label.append(word.split("<!>")[1])
                else:
                    new_label_words_per_label.append(prefix + word)
            new_label_words.append(new_label_words_per_label)
        return new_label_words

    def generate_parameters(self) -> List:
        r"""In basic manual template, the parameters are generated from label words directly.
        In this implementation, the label_words should not be tokenized into more than one token.
        """
        # 处理verbalizer的文本，得到一个张量，内容为文本的token id，维度为(标签数量,标签的最大词组数量, 词组的最大单词数量)，也就是(39,1,9)

        # 处理文本
        all_ids = []
        for words_per_label in self.label_words:
            ids_per_label = []
            for word in words_per_label:
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                ids_per_label.append(ids)
            all_ids.append(ids_per_label)

        # 计算max len 与 max num
        max_len  = max([max([len(ids) for ids in ids_per_label]) for ids_per_label in all_ids])
        max_num_label_words = max([len(ids_per_label) for ids_per_label in all_ids])
        # print(max_len)
        # print(max_num_label_words)
        # max_len指的是token化的label words的最大长度，用于padding,这里目前是9，
        # max_num_label_words指的是每个classes对应的label words的最大数量，这里目前是1.
        words_ids_mask = torch.zeros(max_num_label_words, max_len)
        # (1,9)

        # padding
        words_ids_mask = [[[1]*len(ids) + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]
        # print(words_ids_mask)
        words_ids = [[ids + [0]*(max_len-len(ids)) for ids in ids_per_label]
                             + [[0]*max_len]*(max_num_label_words-len(ids_per_label))
                             for ids_per_label in all_ids]

        # print(words_ids)
        # exit()

        # 转tensor
        words_ids_tensor = torch.tensor(words_ids)
        words_ids_mask = torch.tensor(words_ids_mask)
        self.label_words_ids = nn.Parameter(words_ids_tensor, requires_grad=False)
        # 为避免混淆，使用phrase表示词组，每个标签可以有若干个词组，每个词组里面可以有若干个单词，
        # 这里label_words_ids的维度是(标签数量,标签的最大词组数量, 词组的最大单词数量)，也就是(39,1,9)
        self.words_ids_mask = nn.Parameter(words_ids_mask, requires_grad=False) # A 3-d mask
        self.label_words_mask = nn.Parameter(torch.clamp(words_ids_mask.sum(dim=-1), max=1), requires_grad=False)
        # words_ids_mask.sum(dim=-1)，对最后一维单个label_word的token长度的mask进行sum，得到每个label_word的有效长度。
        # 会使sum的那一维坍缩消失。(39,1)

        # 对于nn.Parameter，首先可以把这个函数理解为类型转换函数，将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并将这个parameter绑定到这个module里面(net.parameter()中就有这个绑定的parameter，所以在参数优化的时候可以进行优化的)，
        # 所以经过类型转换这个self.v变成了模型的一部分，成为了模型中根据训练可以改动的参数了。
        # 使用这个函数的目的也是想让某些变量在学习的过程中不断的修改其值以达到最优化。
        # Tensor相当于是一个高维度的矩阵，它是Variable类的子类。Variable和Parameter之间的差异体现在与Module关联时。
        # 当Parameter作为model的属性与module相关联时，它会被自动添加到Parameters列表中，并且可以使用net.Parameters()迭代器进行访问。
        # requires_grad是Pytorch中通用数据结构Tensor的一个属性，用于说明当前量是否需要在计算中保留对应的梯度信息，


    def project(self,
                logits: torch.Tensor,
                **kwargs,
                ) -> torch.Tensor:
        r"""
        Project the labels, the return value is the normalized (sum to 1) probs of label words.
        normalize 称为归一化

        Args:
            logits (:obj:`torch.Tensor`): The original logits of label words.

        Returns:
            :obj:`torch.Tensor`: The normalized logits of label words
        """
        # 这里输入的logits的shape是(bsz, vocab_size)，也就是output_at_mask
        # print(logits.shape)
        # (8,29886)
        label_words_logits = logits[:, self.label_words_ids]
        # print(self.label_words_ids)
        # TODO 搞清楚这个操作是什么实现的，logits的shape是(8,29886)，label_words_ids的shape是 (39,1,9) 这个是tensor作为索引，并且是多维tensor作为索引。
        ## 这相当于对label_words_logits[1]作为源表进行查询。也就是对1维tensor，使用多维tensor作为索引进行查询。
        ## 如果是对多维tensor索引，比如src tensor是2维，index tensor形状随意，但值不能越界，src_tensor[index_tensor]是对第0个维度[0]索引，
        ## （如果需要对后面的维度索引，例如对第1个维度[1]索引，则表达式改成src_tensor[:,index_tensor]）
        ## 操作可以看成，在index tensor，将索引值换成其在src tensor中的目标值即可，在对1维tensor索引时，该值为标量，也就是src[index],
        ## 在对多维tensor索引时，该值为矩阵,具体为src[index,:]维度比src tensor少1维。
        ## 如果是使用src_tensor[:,index_tensor],则应该在index_tensor中将index换成src[:,index,:]

        # logits的shape是(8,29886)，看成是8个(29886,)的tensor，相当于对其索引八次，最后将结果合到一起就行。
        # label_words_ids要求是long，bool等类型的tensor，作为索引矩阵，取logits中的值，
        # 也就是，其值就是索引，根据这个索引在logits中查表，将查表所得的值，按照label_words_ids的shape生成新的tensor，shape是(39,1,9)
        # 重复8次，并将8个结果矩阵合并，得到最终的结果矩阵，shape是(8,39,1,9)
        ## label_words_logits的shape是(8,39,1,9)
        # print(label_words_logits.shape)

        label_words_logits = self.handle_multi_token(label_words_logits, self.words_ids_mask)
        # 对label_words_logits的最后一维进行降维，降维之后的shape是(8,39,1)
        label_words_logits -= 10000*(1-self.label_words_mask)
        # mask是因为不是每个class都达到了max_num_label_words，也就是说，有的class的label words可能有2个，有的有3个，
        # 需要padding到统一数量，则需要一个mask矩阵记录哪些位置是padding。
        # 然后通过这个操作，给padding位置的logits赋低值，避免模型预测出错。

        # print(label_words_logits.shape)
        # (8,39,1)
        # exit()
        return label_words_logits

    def process_logits(self, logits: torch.Tensor, **kwargs):
        r"""A whole framework to process the original logits over the vocabulary, which contains four steps:

        (1) Project the logits into logits of label words

        if self.post_log_softmax is True:

            (2) Normalize over all label words

            (3) Calibrate (optional)

        (4) Aggregate (for multiple label words)

        Args:
            logits (:obj:`torch.Tensor`): The original logits.

        Returns:
            (:obj:`torch.Tensor`): The final processed logits over the labels (classes).
        """
        # project the logits into logits of label words。
        # 这里输入的logits的shape是(bsz, vocab_size)，也就是output_at_mask，
        # 输出的label_words_logit的shape是(bsz, class_num, max_num_label_words)
        label_words_logits = self.project(logits, **kwargs)
        #Output: (batch_size, num_classes) or  (batch_size, num_classes, max_num_label_words_per_label)
        # 如果max_num_label_words_per_label=1，则是第一种output，否则是第二种。

        if self.post_log_softmax:
            # normalize
            # 归一化，主要是将logits规范在0~1之间并且总和为1，一般都是使用softmax
            label_words_probs = self.normalize(label_words_logits)

            # calibrate
            # 校准，主要是debias
            if  hasattr(self, "_calibrate_logits") and self._calibrate_logits is not None:
                label_words_probs = self.calibrate(label_words_probs=label_words_probs)
            # TODO 需要确认下默认配置下是否有进行校准。之后应该不会使用

            # convert to logits
            label_words_logits = torch.log(label_words_probs+1e-15)

        # aggregate
        # 聚合
        label_logits = self.aggregate(label_words_logits)
        # (bsz, class_num) (8,39)
        return label_logits

    def normalize(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Given logits regarding the entire vocabulary, return the probs over the label words set.

        Args:
            logits (:obj:`Tensor`): The logits over the entire vocabulary.

        Returns:
            :obj:`Tensor`: The logits over the label words set.

        """
        batch_size = logits.shape[0]
        return F.softmax(logits.reshape(batch_size, -1), dim=-1).reshape(*logits.shape)


    def aggregate(self, label_words_logits: torch.Tensor) -> torch.Tensor:
        r"""Use weight to aggregate the logits of label words.

        Args:
            label_words_logits(:obj:`torch.Tensor`): The logits of the label words.

        Returns:
            :obj:`torch.Tensor`: The aggregated logits from the label words.
        """
        # 聚合，因为部分class对应了若干个label words，每个label words都有自己的logits，因此需要聚合，也就是压缩维度，将(8,39,1)压缩为(8,39)
        # todo 这个logits是怎么计算的，用来聚合单个标签所有词组的logits。
        label_words_logits = (label_words_logits * self.label_words_mask).sum(-1)/self.label_words_mask.sum(-1)
        # print(label_words_logits.shape)
        # (8,39)
        # print(self.label_words_mask.shape)
        # (39,1)
        # exit()
        return label_words_logits

    def calibrate(self, label_words_probs: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""

        Args:
            label_words_probs (:obj:`torch.Tensor`): The probability distribution of the label words with the shape of [``batch_size``, ``num_classes``, ``num_label_words_per_class``]

        Returns:
            :obj:`torch.Tensor`: The calibrated probability of label words.
        """
        shape = label_words_probs.shape
        assert self._calibrate_logits.dim() ==  1, "self._calibrate_logits are not 1-d tensor"
        calibrate_label_words_probs = self.normalize(self.project(self._calibrate_logits.unsqueeze(0), **kwargs))
        assert calibrate_label_words_probs.shape[1:] == label_words_probs.shape[1:] \
             and calibrate_label_words_probs.shape[0]==1, "shape not match"
        label_words_probs /= (calibrate_label_words_probs+1e-15)
        # normalize # TODO Test the performance
        norm = label_words_probs.reshape(shape[0], -1).sum(dim=-1,keepdim=True) # TODO Test the performance of detaching()
        label_words_probs = label_words_probs.reshape(shape[0], -1) / norm
        label_words_probs = label_words_probs.reshape(*shape)
        return label_words_probs










