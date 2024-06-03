# coding=utf-8
import os


class Config(object):
    def __init__(self, dataset, domain):
        assert dataset in ["SNIPS", "ATIS", "MultiWOZ", "SGD"]
        path = os.path.join('data', dataset)

        self.seg_label_file = os.path.join(path, 'seg_label.txt')
        self.slot_label_file = os.path.join(path, 'slot_label.txt')

        self.train_file = os.path.join(path, domain, 'train.tsv')
        self.dev_file = os.path.join(path, domain, 'dev.tsv')
        self.test_file = os.path.join(path, domain, 'test.tsv')

        # max len : atis 为46， snips为34, multiwoz 为43, sgd 为84
        # atis 大概6k个utt，dev+test = 200~400, multiwoz 大概7w，dev+test= 6k, sgd大概17w,dev+test大概2w
        self.max_length = 128
        # 对于其他三个数据集选用64，对于SGD选用128
        self.use_cuda = True
        self.gpu = 0

        # TODO 之后考虑如何加lr decay，以及梯度裁剪
        # self.lr_decay = 0.00001

        self.dropout = 0.1
        self.weight_decay = 0.01
        self.batch_size = 64
        self.early_stop = 5

        self.lr = 5e-5
        self.optim = 'AdamW'
        self.checkpoint = 'result/'
        self.base_epoch = 50

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])
