# coding=utf-8
import os.path
import time
from seqeval.metrics import classification_report
import torch
from tqdm import tqdm
from config import Config
from model import BERT_CRF
from transformers import logging
from utils import load_vocab, read_corpus, load_model, save_model, load_data, set_seed
from torchcrf import CRF
import torch.nn as nn
from transformers import BertModel

logging.set_verbosity_error()


def train(dataset, domain, **kwargs):
    config = Config(dataset, domain)
    config.update(**kwargs)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)
    seg_label_dic = load_vocab(config.seg_label_file)
    seg_size = len(seg_label_dic)
    slot_label_dic = load_vocab(config.slot_label_file)
    slot_size = len(slot_label_dic)

    # region 读取数据，并载入dataloader
    train_data = read_corpus(config.train_file, max_length=config.max_length, seg_label_dic=seg_label_dic,
                             slot_label_dic=slot_label_dic)
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, seg_label_dic=seg_label_dic,
                           slot_label_dic=slot_label_dic)
    train_loader = load_data(train_data, shuffle=True, batch_size=config.batch_size)
    dev_loader = load_data(dev_data, shuffle=False, batch_size=config.batch_size)
    # endregion

    model = BERT_CRF(seg_size, slot_size, dropout=config.dropout)
    if config.use_cuda:
        model.cuda()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    best_dev_f1 = 0
    early_stop = config.early_stop
    no_improvement_num = 0

    print(f"\n{dataset}: {domain}")

    # 训练过程
    for epoch in range(config.base_epoch):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, batch in pbar:
            pbar.set_description(f'Dataset: {dataset}, Domain: {domain}, Epoch: {epoch}')
            inputs, masks, seg_tags, slot_tags = batch
            if config.use_cuda:
                inputs, masks, seg_tags, slot_tags = inputs.cuda(), masks.cuda(), seg_tags.cuda(), slot_tags.cuda()
            seg_feats, slot_feats = model(inputs, masks)

            loss = model.loss(seg_feats, slot_feats, seg_tags, slot_tags, masks)
            pbar.set_postfix(loss=round(loss.item(), 4))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_f1, _ = eval(model, dev_loader, epoch, config, seg_label_dic)

        if dev_f1 > best_dev_f1:
            save_model(model, dataset=dataset, domain=domain, dev_f1=dev_f1, split='dev')
            best_dev_f1 = dev_f1
            no_improvement_num = 0
        else:
            no_improvement_num += 1
            print(f'No Improvement: {no_improvement_num}')
        if no_improvement_num >= early_stop:
            break


def eval(model, dev_loader, epoch, config, seg_label_dic):
    all_pseudos = []
    all_predictions = []
    all_seg_tags = []
    valid_lens_list = []

    # eval
    with torch.no_grad():
        model.eval()
        for i, batch in enumerate(dev_loader):
            inputs, masks, seg_tags, slot_tags = batch
            all_seg_tags.extend(seg_tags)
            if config.use_cuda:
                inputs, masks = inputs.cuda(), masks.cuda()
            seg_feats, _ = model(inputs, masks)
            # 因为计算的时候输入的masks，因此输出的预测结果是有效长度的。
            predictions = model.seg_crf.decode(seg_feats, masks)
            all_predictions.extend(predictions)
            # 此处的predictions是list，元素不是tensor，可以直接输出
            all_pseudos.extend(predictions)
            valid_lens = masks.sum(1).tolist()
            valid_lens_list.extend(valid_lens)
    torch.cuda.empty_cache()

    id2seg_label = {v: k for k, v in seg_label_dic.items()}

    trim_preds = []
    trim_trues = []
    for idx, (pred, true) in enumerate(zip(all_predictions, all_seg_tags)):
        valid_pred = pred
        valid_true = true.tolist()

        trim_pred = []
        trim_true = []
        for valid_pred_id, valid_true_id in zip(valid_pred, valid_true):
            if id2seg_label[valid_true_id] in ['X', '<eos>', '<pad>', '<start>']:
                continue
            if id2seg_label[valid_pred_id] in ['X', '<eos>', '<pad>', '<start>']:
                trim_pred.append("O")
                trim_true.append(id2seg_label[valid_true_id])
            else:
                trim_pred.append(id2seg_label[valid_pred_id])
                trim_true.append(id2seg_label[valid_true_id])
        trim_preds.append(trim_pred)
        trim_trues.append(trim_true)

    report = classification_report(trim_trues, trim_preds, digits=4, output_dict=True)
    micro_f1 = report['micro avg']['f1-score']

    print('Eval  Epoch: {} | Micro F1 Scores: {:.4f}'.format(epoch, micro_f1))
    model.train()
    return micro_f1, trim_preds


def test(dataset, domain, ckpt):
    config = Config(dataset, domain)
    seg_label_dic = load_vocab(config.seg_label_file)
    seg_size = len(seg_label_dic)
    slot_label_dic = load_vocab(config.slot_label_file)
    slot_size = len(slot_label_dic)

    model = BERT_CRF(seg_size, slot_size, dropout=config.dropout)
    model = load_model(model, ckpt)
    model.cuda()

    # region 数据读取与载入
    dev_data = read_corpus(config.dev_file, max_length=config.max_length, seg_label_dic=seg_label_dic,
                           slot_label_dic=slot_label_dic)
    dev_loader = load_data(dev_data, shuffle=False, batch_size=config.batch_size)
    # start = time.time()

    test_data = read_corpus(config.test_file, max_length=config.max_length, seg_label_dic=seg_label_dic,
                            slot_label_dic=slot_label_dic)
    test_loader = load_data(test_data, shuffle=False, batch_size=config.batch_size)

    # endregion

    dev_f1, trim_prds = eval(model, dev_loader, -1, config, seg_label_dic)
    gen_pseudos_data(dataset, dev_data, trim_prds, domain=domain, split='dev')

    test_f1, trim_preds = eval(model, test_loader, -1, config, seg_label_dic)
    # end = time.time()
    # print(f'程序运行时间:%.2f秒' % (end - start))

    gen_pseudos_data(dataset, test_data, trim_preds, domain=domain, split='test')


def gen_pseudos_data(dataset, data, trim_preds, domain, split):
    lines = []
    for idx, pred in enumerate(trim_preds):
        raw_line = data[idx].raw_line.strip()
        x = raw_line.strip().split('\t')[0].split()
        assert len(x) == len(pred)
        pred_line = ' '.join(pred)
        line = '\t'.join([raw_line, pred_line])
        lines.append(line)

    # 先只给test做伪标签
    path = os.path.join('data/pseudo', dataset, domain)
    os.makedirs(path) if not os.path.exists(path) else path
    name = split + '.tsv'
    file = os.path.join(path, name)
    with open(file, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print(f'{dataset}: Pseudos label in {domain} for {split} data has been already generated')


if __name__ == '__main__':

    datasets = ["SNIPS", "ATIS", "MultiWOZ", "SGD"]
    dataset2domains = {
        "SNIPS": ['AddToPlaylist', "BookRestaurant", "GetWeather", "PlayMusic",
                  "RateBook", "SearchCreativeWork", "SearchScreeningEvent", "Std"],
        "ATIS": ["Abbreviation", "Airfare", "Airline", "Flight",
                 "GroundService", "Others", "Std"],
        "MultiWOZ": ["BookHotel", "BookRestaurant", "BookTrain", "FindAttraction", "FindHotel",
                     "FindRestaurant", "FindTaxi", "FindTrain", "Others", "Std"],
        "SGD": ["Buses", "Calendar", "Events", "Flights", 'Homes', "Hotels", "Movies", "Music",
                "RentalCars", "Restaurants", "RideSharing", "Services", "Travel", "Weather", "Others"],
    }
    sample_num_list = ['0', '20', '50']

    mode = 'train'
    # mode = 'test'

    # train
    if mode == 'train':
        for dataset in datasets[0:1]:
            for domain in dataset2domains[dataset][:]:
                if domain == "Std":
                    train(dataset=dataset, domain=domain)
                elif dataset not in ["SNIPS", "MultiWOZ"]:
                    train(dataset=dataset, domain=domain)
                else:
                    for sample_num in sample_num_list[0:1]:
                        train(dataset=dataset, domain=domain + '_' + sample_num)

    # test
    elif mode == 'test':
        for dataset in datasets[0:3]:
            print(dataset)
            path = os.path.join('result', dataset, 'dev')
            ckpts = os.listdir(path)
            ckpts.sort()
            # print(ckpts)
            for ckpt in ckpts:

                if ckpt.split('_')[1] == 'Std':
                    domain = ckpt.split('_')[0]
                    print(domain)
                    file = os.path.join(path, ckpt)
                    test(dataset, domain, file)
                elif dataset not in ["SNIPS", "MultiWOZ"]:
                    domain = ckpt.split('_')[0]
                    print(domain)
                    file = os.path.join(path, ckpt)
                    test(dataset, domain, file)
                else:
                    for sample_num in sample_num_list[0:1]:
                        domain = ckpt.split('_')[0] + '_' + ckpt.split('_')[1]
                        print(domain)
                        file = os.path.join(path, ckpt)
                        test(dataset, domain, file)


