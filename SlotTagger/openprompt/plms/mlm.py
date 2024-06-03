
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name

from openprompt.plms.utils import TokenizerWrapper
from typing import List, Dict
from collections import defaultdict

# MLM，举例，BERT、ERNIE（中文NLP预训练模型）
class MLMTokenizerWrapper(TokenizerWrapper):
    add_input_keys = ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def mask_token_ids(self):
        return self.tokenizer.mask_token_id

    @property
    def num_special_tokens_to_add(self):
        if not hasattr(self, '_num_specials'):
            self._num_specials = self.tokenizer.num_special_tokens_to_add()
        return self._num_specials

    def tokenize_one_example(self, wrapped_example, teacher_forcing):
        ''' # TODO doesn't consider the situation that input has two parts
        '''

        wrapped_example, others = wrapped_example

        # for some dataset like SuperGLUE.COPA, the answer requires prediction an span of
        # the input. Or in generation tasks, we need to generate a piece of target_text.
        # In these case, it tokenized to the encoded_tgt_text for future use.
        # 看样子好像是把可能的span of input，提前处理好放入tgt_text。
        encoded_tgt_text = []
        if 'tgt_text' in others:
            tgt_text = others['tgt_text']
            if isinstance(tgt_text, str):
                tgt_text = [tgt_text]
            for t in tgt_text:
                encoded_tgt_text.append(self.tokenizer.encode(t, add_special_tokens=False))


        mask_id = 0 # the i-th the mask token in the template.

        #  defaultdict(list),会构建一个默认value为list的字典，
        encoder_inputs = defaultdict(list)

        for piece in wrapped_example:
            if piece['loss_ids']==1:
                if teacher_forcing: # fill the mask with the tgt task
                    raise RuntimeError("Masked Language Model can't perform teacher forcing training!")
                else:
                    encode_text = [self.mask_token_ids]
                mask_id += 1

            if piece['text'] in self.special_tokens_maps.keys():
                to_replace = self.special_tokens_maps[piece['text']]
                if to_replace is not None:
                    piece['text'] = to_replace
                else:
                    raise KeyError("This tokenizer doesn't specify {} token.".format(piece['text']))

            if 'soft_token_ids' in piece and piece['soft_token_ids']!=0:
                encode_text = [0] # can be replace by any token, since these token will use their own embeddings
            else:
                encode_text = self.tokenizer.encode(piece['text'], add_special_tokens=False)

            encoding_length = len(encode_text)
            encoder_inputs['input_ids'].append(encode_text)
            for key in piece:
                if key not in ['text']:
                    encoder_inputs[key].append([piece[key]]*encoding_length)

        # 如果超过预设的最大长度，则会被截断，应该避免被截断，global max seq len 应该为128至少
        encoder_inputs = self.truncate(encoder_inputs=encoder_inputs)

        # delete shortenable ids
        encoder_inputs.pop("shortenable_ids")
        encoder_inputs = self.concate_parts(input_dict=encoder_inputs)
        encoder_inputs = self.add_special_tokens(encoder_inputs=encoder_inputs)
        # create special input ids
        encoder_inputs['attention_mask'] = [1] *len(encoder_inputs['input_ids'])
        if self.create_token_type_ids:
            encoder_inputs['token_type_ids'] = [0] *len(encoder_inputs['input_ids'])
        # padding

        # todo 将padding部分改进到collate fn中，而非在此处的tokenizer中
        encoder_inputs = self.padding(input_dict=encoder_inputs, max_len=self.max_seq_length, pad_id_for_inputs=self.tokenizer.pad_token_id)

        # convert defaultdict to dict
        if len(encoded_tgt_text) > 0:
            encoder_inputs = {**encoder_inputs, "encoded_tgt_text": encoded_tgt_text}
        else:
            encoder_inputs = {**encoder_inputs}
        return encoder_inputs











