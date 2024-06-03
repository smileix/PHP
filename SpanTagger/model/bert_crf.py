# coding=utf-8
from torchcrf import CRF
import torch.nn as nn
from transformers import BertModel, RobertaModel

class BERT_CRF(nn.Module):
    """
    bert_lstm_crf model
    """
    def __init__(self, seg_size, slot_size, dropout, embedding_dim=768):
        super(BERT_CRF, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir='./.cache')
        # self.bert = RobertaModel.from_pretrained("/Work18/2020/weixiao/code/model/roberta/roberta_base")

        self.dropout = nn.Dropout(p=dropout)
        self.seg_linear = nn.Linear(embedding_dim, seg_size)
        self.slot_linear = nn.Linear(embedding_dim, slot_size)
        self.seg_crf = CRF(num_tags=seg_size, batch_first=True)
        self.slot_crf = CRF(num_tags=slot_size, batch_first=True)
        self.alpha = 0.5

    def forward(self, inputs, attention_mask=None):
        """
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        """
        batch_size = inputs.size(0)
        seq_length = inputs.size(1)
        # batch_size = len(inputs)
        # seq_length = len(inputs[0])

        bert_out, _ = self.bert(inputs, attention_mask=attention_mask, return_dict=False)

        # lstm_out, hidden = self.lstm(embeds)
        # bert_out = embeds.contiguous().view(-1, self.hidden_dim*2)
        d_bert_out = self.dropout(bert_out)
        seg_out = self.seg_linear(d_bert_out)
        slot_out = self.slot_linear(d_bert_out)

        seg_feats = seg_out.contiguous().view(batch_size, seq_length, -1)
        slot_feats = slot_out.contiguous().view(batch_size, seq_length, -1)
        return seg_feats, slot_feats

    def loss(self, seg_feats, slot_feats, seg_tags, slot_tags, mask):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        seg_loss_value = 0 - self.seg_crf(seg_feats, seg_tags, mask, reduction="mean")
        slot_loss_value = 0 - self.slot_crf(slot_feats, slot_tags, mask, reduction='mean')
        loss_value = seg_loss_value + self.alpha * slot_loss_value
        return loss_value



