# coding=utf-8
import copy
import torch
import os
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import *
import random
import numpy as np
from transformers import BertTokenizer, RobertaTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# tokenizer = RobertaTokenizer.from_pretrained("/Work18/2020/weixiao/code/model/roberta/roberta_base")


class InputFeatures(object):
    def __init__(self, idx, raw_line, input_id, seg_label_id, input_mask, slot_label_id):
        self.idx = idx
        self.raw_line = raw_line
        self.input_id = input_id
        self.seg_label_id = seg_label_id
        self.input_mask = input_mask
        self.slot_label_id = slot_label_id


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def read_corpus(path, max_length, seg_label_dic, slot_label_dic):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param seg_label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = []
    # max_l = 0
    for idx, line in enumerate(content):
        # todo 之后如果需要用到domain label的话，也可以添加上这部分信息，或者说先添加上，用不到就放着不用就行。后续可以实践一下，添加domain任务。
        raw_text, raw_label, domain_label = line.strip().split('\t')
        raw_tokens = raw_text.split()
        label = raw_label.split()

        raw_seg_label = []
        raw_slot_label = []
        for temp in label:
            seg = temp[0]
            if seg == 'O':
                slot = 'O'
            else:
                slot = '-'.join(temp.split('-')[1:])
            raw_seg_label.append(seg)
            raw_slot_label.append(slot)

        # 先使用tokenizer，之后再转成token id
        tokens = []
        seg_label = []
        slot_label = []
        for i, word in enumerate(raw_tokens):
            subwords = tokenizer.tokenize(word)
            tokens.extend(subwords)
            # 处理label的时候，也可以直接顺便转成id，这里是用的文本
            for j in range(len(subwords)):
                if j == 0:
                    seg_label.append(raw_seg_label[i])
                    slot_label.append(raw_slot_label[i])
                else:
                    seg_label.append("X")
                    slot_label.append("X")

        # print(label)
        # if len(tokens) > max_length - 2:
        #     tokens = tokens[0:(max_length - 2)]
        #     seg_label = seg_label[0:(max_length - 2)]

        input_ids = tokenizer.convert_tokens_to_ids([tokenizer.cls_token] + tokens + [tokenizer.sep_token])
        seg_label_f = ["<start>"] + seg_label + ['<eos>']
        slot_label_f = ["<start>"] + slot_label + ['<eos>']
        seg_label_ids = [seg_label_dic[i] for i in seg_label_f]
        slot_label_ids = [slot_label_dic[i] for i in slot_label_f]
        input_mask = [1] * len(input_ids)
        assert len(seg_label_ids) == len(input_ids)
        assert len(slot_label_ids) == len(input_ids)

        # 此处的raw text与raw label都是str，并没有按照空格分割成list，方便后续使用处理。
        feature = InputFeatures(idx=idx, raw_line=line, input_id=input_ids, input_mask=input_mask,
                                seg_label_id=seg_label_ids, slot_label_id=slot_label_ids)
        result.append(feature)


    return result


class MyDataset(Dataset):
    def __init__(self, split_ids, split_masks, split_seg_tags, split_slot_tags):
        self.split_ids = split_ids
        self.split_masks = split_masks
        self.split_seg_tags = split_seg_tags
        self.split_slot_tags = split_slot_tags

    def __len__(self):
        return len(self.split_ids)

    def __getitem__(self, item):
        return [self.split_ids[item], self.split_masks[item], self.split_seg_tags[item], self.split_seg_tags[item]]

    @staticmethod
    def collate_fn(batch):
        # batch 是个list，长度为64，也就是bsz
        # batch中的每个元素是个长度为4的list
        # 先遍历一遍找到最大值
        batch_max_len = 0
        for sample in batch:
            seq_len = len(sample[0])
            batch_max_len = max(seq_len, batch_max_len)
        # print(batch_max_len)

        batch_inputs = []
        batch_masks = []
        batch_seg_tags = []
        batch_slot_tags = []

        for sample in batch:
            sample_input = copy.deepcopy(sample[0])
            sample_masks = copy.deepcopy(sample[1])
            sample_seg_tag = copy.deepcopy(sample[2])
            sample_slot_tag = copy.deepcopy(sample[3])

            while len(sample_input) < batch_max_len:
                sample_input.append(0)
                sample_masks.append(0)
                sample_seg_tag.append(0)
                sample_slot_tag.append(0)


            assert len(sample_input) == len(sample_masks) == len(sample_seg_tag) == len(sample_slot_tag) == batch_max_len

            batch_inputs.append(sample_input)
            batch_masks.append(sample_masks)
            batch_seg_tags.append(sample_seg_tag)
            batch_slot_tags.append(sample_slot_tag)

        batch_inputs_tensor = torch.LongTensor(batch_inputs)
        batch_masks_tensor = torch.tensor(batch_masks, dtype=torch.bool)
        batch_seg_tags_tensor = torch.LongTensor(batch_seg_tags)
        batch_slot_tags_tensor = torch.LongTensor(batch_slot_tags)

        return batch_inputs_tensor, batch_masks_tensor, batch_seg_tags_tensor, batch_slot_tags_tensor


def load_data(data, shuffle, batch_size):
    split_ids = [temp.input_id for temp in data]
    split_masks = [temp.input_mask for temp in data]
    split_seg_tags = [temp.seg_label_id for temp in data]
    split_slot_tags = [temp.slot_label_id for temp in data]
    split_dataset = MyDataset(split_ids, split_masks, split_seg_tags, split_slot_tags)
    split_loader = DataLoader(split_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=MyDataset.collate_fn)
    return split_loader


def save_model(model, **kwargs):
    path = os.path.join('result', kwargs.get('dataset'))
    os.mkdir(path) if not os.path.exists(path) else path
    name = kwargs.get('domain') + '_' + str(round((kwargs['dev_f1']), 4)) + '_' + kwargs.get(
        'split') + '.pt'
    full_name = os.path.join(path, name)
    torch.save(model.state_dict(), full_name)
    print('Saved model successfully')


def load_model(model, file='result', **kwargs):
    assert os.path.exists(file)
    model.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(file))
    return model


def set_seed(seed: Optional[int] = None):
    """set seed for reproducibility
    Args:
        seed (:obj:`int`): the seed to seed everything for reproducibility. if None, do nothing.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Global seed set to {seed}")
