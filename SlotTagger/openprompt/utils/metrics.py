from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from typing import *
from openprompt.utils.logging import logger
from seqeval.metrics import classification_report
from openprompt.data_info import unseen_slot_based_dataset_domain
import seqeval

def f1(p, r):
    if p == 0. or r == 0.:
        return 0.
    return 2*p*r/(p+r)

def label_path(label, label_path_sep):
    label = label.strip(label_path_sep)
    # 去掉开头跟结尾的连字符
    label_path = label.split(label_path_sep)
    label_set = []
    for i in range(len(label_path)):
        label_set.append(label_path_sep.join(label_path[:i+1]))
    return label_set

def loose_micro(labels, preds, id2label, label_path_sep):
    if id2label is None:
        raise ValueError("no id2label dict provided, cannot calculate loose_micro_f1 !")
    labels = [label_path(id2label[i], label_path_sep) for i in labels]
    preds = [label_path(id2label[i], label_path_sep) for i in preds]
    cnt_pred = 0
    cnt_label = 0
    cnt_correct = 0
    for label, pred in zip(labels, preds):
        label = set(label)
        pred = set(pred)
        cnt_pred += len(pred)
        cnt_label += len(label)
        cnt_correct += len(label.intersection(pred))
    p = cnt_correct/cnt_pred
    r = cnt_correct/cnt_label
    f = f1(p, r)
    return {'precision': p, 'recall': r, 'f1': f}

def loose_macro(labels, preds, id2label, label_path_sep):
    if id2label is None:
        raise ValueError("no id2label dict provided, cannot calculate loose_micro_f1 !")
    labels = [label_path(id2label[i], label_path_sep) for i in labels]
    preds = [label_path(id2label[i], label_path_sep) for i in preds]
    p = 0.
    r = 0.
    for label, pred in zip(labels, preds):
        label = set(label)
        pred = set(pred)
        if len(pred) > 0:
            p += len(label.intersection(pred))/len(pred)
        if len(label) > 0:
            r += len(label.intersection(pred))/len(label)
    p /= len(labels)
    r /= len(labels)
    f = f1(p, r)
    return {'precision': p, 'recall': r, 'f1': f}

def calc_seen_unseen_slot_f1(all_golds, all_preds, dataset, domain):
    unseen_slots = unseen_slot_based_dataset_domain[dataset][domain]
    all_seen_preds = []  # 用来存seen_slot部分的预测
    all_seen_golds = []  # 用来存seen_slot部分的label
    all_unseen_preds = []  # 用来存unseen_slot部分的预测
    all_unseen_golds = []  # 用来存unseen_slot部分的label

    for preds, golds in zip(all_preds, all_golds):
        seen_preds = []
        seen_golds = []
        unseen_preds = []
        unseen_golds = []
        for pred, gold in zip(preds, golds):
            if pred != 'O' and pred[2:] not in unseen_slots:  # 如果预测的是seen slot
                pred_seen = pred
            else:
                pred_seen = 'O'
            if gold != 'O' and gold[2:] not in unseen_slots:  # 如果label是seen_slot
                gold_seen = gold
            else:
                gold_seen = 'O'
            seen_preds.append(pred_seen)  # 把seen_pred加进来，如果seen_pred是'O'则不影响结果
            seen_golds.append(gold_seen)

            if pred != 'O' and pred[2:] in unseen_slots:  # 如果预测的是unseen slot
                pred_unseen = pred
            else:
                pred_unseen = 'O'
            if gold != 'O' and gold[2:] in unseen_slots:  # 如果label是unseen_slot
                gold_unseen = gold
            else:
                gold_unseen = 'O'

            unseen_preds.append(pred_unseen)  # 把unseen_pred加进来，如果unseen_pred是'O'则不影响结果
            unseen_golds.append(gold_unseen)

        all_seen_preds.append(seen_preds)
        all_seen_golds.append(seen_golds)
        all_unseen_preds.append(unseen_preds)
        all_unseen_golds.append(unseen_golds)

    print('seen_slots f1:', seqeval.metrics.f1_score(all_seen_golds, all_seen_preds))
    print('unseen_slots f1:', seqeval.metrics.f1_score(all_unseen_golds, all_unseen_preds))

def classification_metrics(preds: Sequence[int],
                           labels: Sequence[int],
                           metric: Optional[str] = "micro-f1",
                           id2label: Optional[Dict] = None,
                           label_path_sep: Optional[str] = '-',
                           dataset = None,
                           domain = None,
                          ) -> float:
    """evaluation metrics for classification task.

    Args:
        preds (Sequence[int]): predicted label ids for each examples
        labels (Sequence[int]): gold label ids for each examples
        metric (str, optional): type of evaluation function, support 'micro-f1', 'macro-f1', 'accuracy', 'precision', 'recall'. Defaults to "micro-f1".

    Returns:
        score (float): evaluation score
    """
    
    if metric == "micro-f1":
        score = f1_score(labels, preds, average='micro')
    elif metric == "seqeval":
        report = classification_report(labels, preds, digits=4, output_dict=True)
        score = report['micro avg']['f1-score']
        # report = classification_report(labels, preds, digits=4)
        # print(report)
        # calc_seen_unseen_slot_f1(labels, preds, dataset, domain)
    # <editor-fold desc="Metric">
    elif metric == "macro-f1":
        score = f1_score(labels, preds, average='macro')
    elif metric == "accuracy":
        score = accuracy_score(labels, preds)
    elif metric == "precision":
        score = precision_score(labels, preds)
    elif metric == "recall":
        score = recall_score(labels, preds)
    # only hierarchical label loose metric is supported TODO naive multilabel ?
    elif metric == 'loose-micro-f1': 
        score = loose_micro(labels, preds, id2label=id2label, label_path_sep=label_path_sep)['f1']
    elif metric == 'loose-macro-f1':
        score = loose_macro(labels, preds, id2label=id2label, label_path_sep=label_path_sep)['f1']
    elif metric == 'loose-micro-precision': 
        score = loose_micro(labels, preds, id2label=id2label, label_path_sep=label_path_sep)['precision']
    elif metric == 'loose-macro-precision':
        score = loose_macro(labels, preds, id2label=id2label, label_path_sep=label_path_sep)['precision']
    elif metric == 'loose-micro-recall': 
        score = loose_micro(labels, preds, id2label=id2label, label_path_sep=label_path_sep)['recall']
    elif metric == 'loose-macro-recall':
        score = loose_macro(labels, preds, id2label=id2label, label_path_sep=label_path_sep)['recall']
    else:
        raise ValueError("'{}' is not a valid evaluation type".format(metric))
    # </editor-fold>
    return score

def generation_metric(hypos,
                      refs, 
                      metric: Optional[str] = "sentence_bleu"):
    r"""Some basic metric function for generation. However, many generation tasks
    has their own evaluation bash scripts.

    Args:
        hypos (:obj:`str`) : the generated sentence.
        refs (:obj:`list(str)`) : the referenced (ground-truth) sentence.
        metric (:obj:`str`, `optional`) : the type of metric option

    Returns:
        score (float): evaluate score
    """
    if metric == "sentence_bleu":
        # a simple criterion to visualize the performance, not rigorous.
        import nltk
        try:
            nltk_path = str(nltk.data.find("tokenizers/punkt"))
            logger.info(f"using nltk from: {nltk_path}")
        except LookupError:
            nltk.download('punkt')

        from nltk.translate.bleu_score import sentence_bleu
        from nltk.tokenize import word_tokenize
        from nltk.translate.bleu_score import SmoothingFunction
        smoothie = SmoothingFunction().method4 # a function for smooth
        scores = []
        
        for ref, hypo in zip(refs, hypos):
            tokenized_rs = []
            ref = ref.split("\n")
            for r in ref:
                tokenized_rs.append(word_tokenize(r))
            hypo = word_tokenize(hypo)
            try:
                sc = sentence_bleu(tokenized_rs, hypo, smoothing_function=smoothie)
            except ValueError: # TODO ZeroDivisionError
                logger.warning("math domain error in bleu, set to 0.0. generated sentence: {}".format(hypo))
                sc = 0.0
            scores.append(sc)
        score = sum(scores)/len(scores)
        return score
    else:
        raise ValueError("'{}' is not a valid metric type.".format(metric))