# <editor-fold desc="import modules">
import os, shutil, copy
import sys
import time

sys.path.append(".")

import torch
from torch import nn
from torch.nn.parallel.data_parallel import DataParallel
from openprompt.utils.cuda import model_to_device
from tensorboardX import SummaryWriter

from tqdm import tqdm
import dill
import warnings

from typing import Callable, Union, Dict

try:
    from typing import OrderedDict
except ImportError:
    from collections import OrderedDict

from openprompt.pipeline_base import PromptForClassification, PromptForGeneration
from openprompt import PromptDataLoader
from openprompt.prompts import *
from openprompt.utils.logging import logger
from openprompt.utils.metrics import classification_metrics, generation_metric
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import Adafactor, AdafactorSchedule

from .data_info import dataset2index, dataset2valid_labels


# </editor-fold>


class BaseRunner(object):
    r"""A base runner for training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use others runner class.
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        model (:obj:`nn.Module`): One ``nn.Module`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """

    def __init__(self, model: PromptForClassification, config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None, ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader

        self.wrap_model()

        self.cur_epoch = 0
        self.best_score = None
        self.global_step = 0
        self.writer = SummaryWriter(os.path.join(self.config.logging.path, 'tensorboard'))
        if not os.path.exists(os.path.join(config.logging.path, 'checkpoints')):
            os.mkdir(os.path.join(config.logging.path, 'checkpoints'))

        self.clean = self.config.train.clean

    def __del__(self):
        if hasattr(self, 'writer'):
            self.writer.close()

    def log(self, name, y, x):
        if self.clean: return  # TODO add more types
        self.writer.add_scalar(name, y, x)

    def set_stop_criterion(self):
        """Total training steps, either controlled by num_training_steps or num_epochs"""
        self.early_stop = self.config.train.early_stop
        if hasattr(self.config.train, "num_training_steps") and self.config.train.num_training_steps is not None:
            if self.config.train.num_epochs is not None:
                logger.warning("num_training_steps set explicitly, num_epochs is not in use.")
            self.num_training_steps = self.config.train.num_training_steps
            self.num_epochs = int(1e8)  # set to a large number
        else:
            if self.config.train.num_epochs is None:
                raise RuntimeError("At least num_training_steps & num_epochs should be specified.")
            self.num_training_steps = self.steps_per_epoch * self.config.train.num_epochs
            self.num_epochs = self.config.train.num_epochs

    @property
    def steps_per_epoch(self) -> int:
        """num of training steps per epoch"""
        batches = len(self.train_dataloader)
        effective_accum = self.config.train.gradient_accumulation_steps
        return (batches // effective_accum)

    def wrap_model(self):
        self.model = model_to_device(self.model, self.config.environment)

    @property
    def inner_model(self):
        return self.model.module if isinstance(self.model, DataParallel) else self.model

    def configure_optimizers(self):
        r"""config the optimizer and scheduler for

        1. model

        2. template

        3. verbalizer(optional)
        """

        optimizers = []
        schedulers = []

        if not self.config.plm.optimize.freeze_para:
            no_decay = self.config.plm.optimize.no_decay
            weight_decay = self.config.plm.optimize.weight_decay
            optimizer_grouped_parameters = [{
                'params': [p for n, p in self.inner_model.plm.named_parameters() if
                           not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay
                }, {
                'params': [p for n, p in self.inner_model.plm.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
                }]

            plm_optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.plm.optimize.lr,
                                  betas=self.config.plm.optimize.betas, eps=self.config.plm.optimize.eps,
                                  no_deprecation_warning=True)
            optimizers.append(plm_optimizer)
            if self.config.plm.optimize.scheduler is not None:
                plm_scheduler = get_linear_schedule_with_warmup(plm_optimizer,
                                                                num_warmup_steps=self.config.plm.optimize.scheduler.num_warmup_steps,
                                                                num_training_steps=self.num_training_steps)
                schedulers.append(plm_scheduler)

        class Dummy:
            pass

        template_config = self.config[self.config.template]
        if hasattr(template_config,
                   "optimize") and template_config.optimize is not None:  # TODO should add optimize config in each yaml
            if not hasattr(self.inner_model.template, "optimize"):
                # using default gradient descent optimizer.
                optimizer_grouped_parameters = [{
                    'params': [p for name, p in self.inner_model.template.named_parameters() if
                               'raw_embedding' not in name]
                    }]
                if template_config.optimize.name.lower() == "adamw":
                    template_optimizer = AdamW(optimizer_grouped_parameters, lr=template_config.optimize.lr,
                                               eps=template_config.optimize.adam_epsilon, no_deprecation_warning=True)
                    optimizers.append(template_optimizer)
                    if hasattr(template_config.optimize,
                               "scheduler") and template_config.optimize.scheduler is not None:
                        template_scheduler = get_linear_schedule_with_warmup(template_optimizer,
                                                                             num_warmup_steps=template_config.optimize.scheduler.num_warmup_steps,
                                                                             num_training_steps=self.num_training_steps)
                        schedulers.append(template_scheduler)
                elif template_config.optimize.name.lower() == "adafactor":
                    template_optimizer = Adafactor(optimizer_grouped_parameters, lr=template_config.optimize.lr,
                                                   weight_decay=1e-5, relative_step=False, scale_parameter=False,
                                                   warmup_init=False)
                    # template_scheduler = AdafactorSchedule(template_optimizer)
                    optimizers.append(template_optimizer)  # schedulers.append(template_scheduler)
                else:
                    raise NotImplementedError("Template Optimizer not Implemented!")
            else:
                template_optimizer = Dummy()
                # resemble a pytorch optimizer for unified training.
                setattr(template_optimizer, "step", self.inner_model.template.optimize)
                setattr(template_optimizer, "zero_grad", lambda: None)
                optimizers.append(template_optimizer)

        if hasattr(self.inner_model, "verbalizer") and self.inner_model.verbalizer:
            verbalizer_config = self.config[self.config.verbalizer]
            if hasattr(verbalizer_config, "optimize") and verbalizer_config.optimize is not None:
                if not hasattr(self.inner_model.verbalizer, "optimize"):
                    # using default gradient descent optimizer.
                    verbalizer_optimizer = AdamW(self.inner_model.verbalizer.parameters(),
                                                 lr=verbalizer_config.optimize.lr, no_deprecation_warning=True)
                    optimizers.append(verbalizer_optimizer)
                    if hasattr(verbalizer_config.optimize,
                               "scheduler") and verbalizer_config.optimize.scheduler is not None:
                        verbalizer_scheduler = get_linear_schedule_with_warmup(verbalizer_optimizer,
                                                                               num_warmup_steps=verbalizer_config.optimize.scheduler.num_warmup_steps,
                                                                               num_training_steps=self.num_training_steps)
                        schedulers.append(verbalizer_scheduler)
                else:
                    verbalizer_optimizer = Dummy()
                    # resemble a pytorch optimizer for unified training.
                    setattr(verbalizer_optimizer, "step", self.inner_model.verbalizer.optimize)
                    setattr(verbalizer_optimizer, "zero_grad", lambda: None)
                    optimizers.append(verbalizer_optimizer)

        self.optimizers = optimizers
        self.schedulers = schedulers

    def checkpoint_path(self, ckpt: str) -> str:
        return os.path.join(os.path.join(self.config.logging.path, "checkpoints"), f'{ckpt}.ckpt')

    def load_checkpoint(self, ckpt: str, load_state=True) -> bool:
        logger.info(f"Loading Checkpoint {self.checkpoint_path(ckpt)}...")
        try:
            state_dict = torch.load(self.checkpoint_path(ckpt), pickle_module=dill, map_location="cpu")
        except FileNotFoundError:
            logger.warning(f"Checkpoint {self.checkpoint_path(ckpt)} not found")
            return False

        # load state to model
        self.model = self.inner_model
        self.model.load_state_dict(state_dict['state_dict'])

        if load_state:
            # load state to optimizers
            for optimizer, op_state in zip(self.optimizers, state_dict['optimizer']):
                if isinstance(optimizer, torch.optim.Optimizer):
                    optimizer.load_state_dict(op_state)
            for scheduler, sc_state in zip(self.schedulers, state_dict['scheduler']):
                if isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
                    with warnings.catch_warnings(record=True):
                        scheduler.load_state_dict(sc_state)

            # load training state
            self.cur_epoch = state_dict['cur_epoch'] + 1
            self.best_score = state_dict['best_score']
            self.global_step = state_dict['global_step']
        logger.info(f"Load Checkpoint finished, the current validation metric: {state_dict['validation_metric']}")
        return True

    def save_checkpoint(self, ckpt: str, save_state=True, extra: dict = {}, copy: str = None):
        if self.clean: return
        logger.info(f"Saving checkpoint {self.config.logging.path}...")
        state_dict = {
            "state_dict": self.inner_model.state_dict(),
            }
        state_dict.update(extra)

        if save_state:
            state_dict["optimizer"] = [opt.state_dict() if isinstance(opt, torch.optim.Optimizer) else None for opt in
                                       self.optimizers]
            with warnings.catch_warnings(record=True):
                state_dict["scheduler"] = [
                    sch.state_dict() if isinstance(sch, torch.optim.lr_scheduler._LRScheduler) else None for sch in
                    self.schedulers]

            state_dict.update({
                "cur_epoch": self.cur_epoch, "best_score": self.best_score, "global_step": self.global_step,
                })
        torch.save(state_dict, self.checkpoint_path(ckpt), pickle_module=dill)
        if copy:
            logger.info(f"Copying checkpoint {self.checkpoint_path(ckpt)} to {self.checkpoint_path(copy)}...")
            shutil.copyfile(self.checkpoint_path(ckpt),
                            self.checkpoint_path(copy))  # logger.info(f"Save Checkpoint finished")

    def save_results(self, split, results: dict):
        if self.clean: return
        for name, values in results.items():
            file_name = os.path.join(self.config.logging.path, f"{split}_{name}.txt")
            with open(file_name, 'w') as fout:
                for value in values:
                    print(value, file=fout)

    def inference_epoch(self, split: str):
        outputs = []
        with torch.no_grad():
            self.model.eval()
            data_loader = self.valid_dataloader if split == 'validation' else self.test_dataloader
            for batch_idx, batch in enumerate(data_loader):
                # if batch_idx % 50 == 0:
                #     torch.cuda.empty_cache()
                batch = self.to_cuda(batch)
                outputs.append(self.inference_step(batch, batch_idx))

        metrics = self.inference_epoch_end(split, outputs)
        for metric_name, metric in metrics.items():
            self.log(f'{split}/{metric_name}', metric, self.cur_epoch)
            logger.info(f"Epoch {self.cur_epoch}, {split} / {metric_name} Performance: {round(metric, 4)}")
        return round(metrics.popitem(last=False)[1],
                     4)  # TODO the first metric is the most important one  # return metrics

    def to_cuda(self, batch_dict):
        for key in batch_dict:
            if key in ["input_ids", "attention_mask", "soft_token_ids", "loss_ids"]:
                batch_dict[key] = batch_dict[key].cuda()
        return batch_dict

    def training_epoch(self, epoch):
        self.model.train()
        self.model.zero_grad()
        total_loss = 0.0
        sum_loss = 0.0
        with tqdm(total=self.steps_per_epoch, desc=f"{self.config.dataset.domain} train epoch: {epoch}") as pbar:
            for batch_idx, batch in enumerate(self.train_dataloader):
                # if batch_idx % 50 == 0:
                #     torch.cuda.empty_cache()
                batch = self.to_cuda(batch)
                loss = self.training_step(batch, batch_idx)

                if self.config.train.gradient_accumulation_steps > 1:
                    loss = loss / self.config.train.gradient_accumulation_steps
                sum_loss += loss.item()
                loss.backward()
                if (batch_idx + 1) % self.config.train.gradient_accumulation_steps == 0:
                    pbar.set_postfix({'loss': sum_loss})
                    self.log('train/loss', sum_loss, self.global_step)
                    # logger.info("{} {}".format(self.inner_model.template.soft_embeds.data.mean().item(),self.global_step))

                    if self.config.train.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.max_grad_norm)

                    for optimizer in self.optimizers:
                        optimizer.step()
                    for scheduler in self.schedulers:
                        scheduler.step()
                    for optimizer in self.optimizers:
                        optimizer.zero_grad()

                    total_loss += sum_loss
                    sum_loss = 0.
                    self.global_step += 1
                    pbar.update(1)
                if self.global_step >= self.num_training_steps:
                    # logger.info(
                    #     f"Training epoch {epoch}, num_steps {self.global_step}, avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
                    return -1  # an indicator of stopping the training
        # logger.info(
        # f"Training epoch {epoch}, num_steps {self.global_step},  avg_loss: {total_loss / self.steps_per_epoch:.4f}, total_loss: {total_loss:.4f}")
        return 1

    def on_fit_start(self):
        """Some initialization works"""
        pass

    def fit(self, ckpt: Optional[str] = None):
        self.set_stop_criterion()
        self.configure_optimizers()

        if ckpt:
            if not self.load_checkpoint(ckpt):
                logger.warning("Train from scratch instead ...")
        if self.cur_epoch == 0:
            self.on_fit_start()

        no_improvement_num = 0
        for self.cur_epoch in range(self.cur_epoch, self.num_epochs):
            continue_training = self.training_epoch(self.cur_epoch)
            score = self.inference_epoch("validation")
            copy = None
            # 原先的保存逻辑是，每个epoch都保存模型，也就是last模型，如果其score达到最高，则再复制一份保存为best模型，吞吐很高，io时间上浪费不少，
            # 但这样可以resume，因为模型保存逻辑中有保存优化器的参数，只不过我不需要这个，因此不保存last模型，优化器的参数其实也可以不保存，但我懒得改了
            # 经验上来看，bert模型，不保存优化器的话，ckpt大概是440M左右的大小，保存优化器的话，基本就是1.2G左右的大小。
            # todo 这里是用＞还是≥
            if self.best_score is None or ((score - self.best_score) > 0) == self.config.checkpoint.higher_better:
                copy = 'best'
                self.best_score = score
                no_improvement_num = 0
                # logger.info(f"find better model, whose f1 scores are {score}")
                self.save_checkpoint('best', extra={"validation_metric": score})
            else:
                no_improvement_num += 1
                logger.info(f'Epoch {self.cur_epoch}, No improvement num: {no_improvement_num}')
            # self.save_checkpoint('last', extra={"validation_metric": score}, copy=copy)
            if continue_training == -1 or no_improvement_num >= self.early_stop:
                logger.info("Stop training by reaching maximum num_training_steps or maximum no_improvement_num")
                break
        return self.best_score

    def test(self, ckpt: Optional[str] = None) -> dict:
        if ckpt:
            # 此处加载ckpt
            if not self.load_checkpoint(ckpt, load_state=False):
                logger.error("Test cannot be performed")
                exit()

        # 计算时间
        # start3 = time.time()
        res = self.inference_epoch("test")
        # end3 = time.time()
        # print(f'测试时间:%.2f秒' % (end3 - start3))

        return res

    def run(self, ckpt: Optional[str] = None) -> dict:
        self.fit(ckpt)
        res = self.test(ckpt=None if self.clean else 'best')
        # logger.info(f'{self.config.dataset.domain}, Test Performance: {res}')
        return res  # 这里还要再读取一次best并测试


class ClassificationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use others runner class.
    This class is specially implemented for classification.
    For generation task, though it can be integrated in this class
    via `task` option, we keep it as another class for simplicity.

    Args:
        model (:obj:`PromptForClassification`): One ``PromptForClassification`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
        loss_function (:obj:`Callable`, optional): The loss function in the training process.
    """

    # Optional[X]等价于Union[X, None]
    def __init__(self, model: PromptForClassification, config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None, loss_function: Optional[Callable] = None,
                 id2label: Optional[Dict] = None, ):
        super().__init__(model=model, config=config, train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader, test_dataloader=test_dataloader, )
        self.loss_function = loss_function if loss_function else self.configure_loss_function()
        self.id2label = id2label
        self.label_path_sep = config.dataset.label_path_sep
        self.dataset = config.dataset.name
        self.domain = config.dataset.domain.split('_')[0]

    def configure_loss_function(self, ):
        r"""config the loss function if it's not passed."""
        if self.config.classification.loss_function == "cross_entropy":
            return torch.nn.CrossEntropyLoss()
        elif self.config.classification.loss_function == "nll_loss":
            # negative log likelihood loss 负对数似然损失函数
            return torch.nn.NLLLoss()
        else:
            raise NotImplementedError

    def inference_step(self, batch, batch_idx):

        batch_span_label, batch_ys, batch_span = self._get_label(batch_label=batch['label'])
        batch_guid = batch["guid"]

        logits = self.model(batch)

        # 这里屏蔽了非tgt domain的label。
        if self.domain == 'Std':
            new_logits = logits
        else:
            tgt_slot_index_list = dataset2index[self.dataset][self.domain]
            import numpy as np
            new_logits = np.empty([logits.shape[0], logits.shape[1]])
            new_logits.fill(float("-inf"))
            for tgt_slot_index in tgt_slot_index_list:
                new_logits[:, tgt_slot_index] = logits.cpu()[:, tgt_slot_index]


        pred = torch.argmax(torch.tensor(new_logits), dim=-1)
        # print(batch)
        outputs = zip(batch_guid, batch_span_label, batch_ys, batch_span, pred.cpu().tolist())
        # 相同的guid的prompt，可能被分到不同的batch里面，需要在推理阶段的最后一起划分，而非在batch内划分
        # 这里的pred只是针对span的pred，还需要根据span填充其他位置，之后获得整句的pred。
        return outputs

    def inference_epoch_end(self, split, outputs):
        all_outputs = []
        # outputs的形式如下
        # outputs = zip(batch_guid, batch_span_label, batch_ys, batch_span, pred)
        # 这里的pred只是针对span的pred，还需要根据span填充其他位置，之后获得整句的pred。

        # 先将所有输出整合到一块，然后构建一个dict，利用dict的特性，按照guid归并。相同guid的句子，属于同一个原始输入。
        for output in outputs:
            all_outputs.extend(output)

        # 按照guid归并
        output_dict = {}
        for sample in all_outputs:
            guid, span_label, ys, span, span_pred = sample[0], sample[1], sample[2], sample[3], sample[4]
            output_dict.setdefault(guid, []).append((span_label, ys, span, span_pred))

        # 将span_pred填充到ys中，获得整句的pred label。ys也就是整句的true label，
        all_true_label = []
        all_pred_label = []
        for guid, output_list in output_dict.items():
            # 先获得整句的true label
            true_label = copy.deepcopy(output_list[0][1])
            all_true_label.append(true_label)

            # pred label与true label只有span位置的label不同，其余位置都是O
            # pred_label = copy.deepcopy(output_list[0][1])
            pred_label = ['O'] * len(true_label)
            for (_, _, span, span_pred) in output_list:
                # 将ys中，span位置处的label换成span_pred， 但span_pred是数字，还需要换成对应的token才行。
                pred_label[span[0]] = 'B-' + self.id2label[span_pred]
                if span[0] != span[1]:
                    # 添加合法性判断，如果标签不合法，出现非法组合，需要赋值为O
                    for i in range(span[0] + 1, span[1] + 1):
                        tmp = 'I-' + self.id2label[span_pred]
                        pred_label[i] = tmp if tmp in dataset2valid_labels[self.dataset] else 'O'

            all_pred_label.append(pred_label)

        self.save_results(split, {
            'preds': all_pred_label, 'labels': all_true_label,
            })

        metrics = OrderedDict()
        for metric_name in self.config.classification.metric:
            metric = classification_metrics(all_pred_label, all_true_label, metric_name, id2label=self.id2label,
                                            label_path_sep=self.label_path_sep, dataset=self.dataset, domain=self.domain)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        logits = self.model(batch)
        # 实质上，是将序列标注任务，改成了单句分类任务。因此logits也就是(bsz, label_num),而非之前序列标注框架下的(bsz,seq_len,label_num)
        # (8,39) 也就是（bsz，label_num)
        # print(batch['label'].shape)
        # (bsz,)

        # 因为prompt改成了单句分类任务，logits的shape中并没有seq len，因此对于label来说不需要padding。
        # 取label ，需要对batch['label']进行操作
        batch_span_label, _, _ = self._get_label(batch_label=batch['label'])
        loss = self.loss_function(logits, torch.tensor(batch_span_label).cuda())
        return loss

    def _get_label(self, batch_label):
        # 该方法主要是对batch['label']这个dict进行格式化取数据，以正常batch的形式输出
        batch_span_label = []
        batch_ys = []
        batch_span = []
        for sample in batch_label:
            # sample是个dict, sample['span_label']是int，sample['ys']是list，sample['span']也是list
            span_label = sample['span_label']
            batch_span_label.append(span_label)
            ys = sample['ys']
            batch_ys.append(ys)
            span = sample['span']
            batch_span.append(span)

        return batch_span_label, batch_ys, batch_span

    def on_fit_start(self):
        """Some initialization works"""
        self.prompt_initialize()

    def prompt_initialize(self):
        verbalizer_config = self.config[self.config.verbalizer]
        template_config = self.config[self.config.template]
        if not hasattr(self.inner_model.verbalizer, "optimize_to_initialize") and not hasattr(self.inner_model.template,
                                                                                              "optimize_to_initialize"):
            return None
        if hasattr(verbalizer_config, "init_using_split"):
            using_split = verbalizer_config.init_using_split
        elif hasattr(template_config, "init_using_split"):
            using_split = template_config.init_using_split
        else:
            using_split = "valid"

        if using_split == "train":
            dataloader = self.train_dataloader
        elif using_split == "valid":
            dataloader = self.valid_dataloader
        else:
            raise NotImplementedError

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Init_using_{}".format(using_split)):
                batch = batch.to("cuda:{}".format(self.config.environment.local_rank)).to_dict()
                logits = self.model(batch)
            if hasattr(self.inner_model.verbalizer, "optimize_to_initialize"):
                self.inner_model.verbalizer.optimize_to_initialize()
            if hasattr(self.inner_model.template, "optimize_to_initialize"):
                self.inner_model.template.optimize_to_initialize()

        self.wrap_model()


class GenerationRunner(BaseRunner):
    r"""A runner for simple training without training tricks.
    Applying training tricks such as ensemble of template or verbalizer,
    or self-training can use others runner class.
    This class is specially implemented for generation.

    Args:
        model (:obj:`PromptForGeneration`): One ``PromptForGeneration`` object.
        train_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the training data.
        valid_dataloader (:obj:`PromptDataloader`, optionla): The dataloader to bachify and process the val data.
        test_dataloader (:obj:`PromptDataloader`, optional): The dataloader to bachify and process the test data.
        config (:obj:`CfgNode`): A configuration object.
    """

    def __init__(self, model: PromptForGeneration, config: CfgNode = None,
                 train_dataloader: Optional[PromptDataLoader] = None,
                 valid_dataloader: Optional[PromptDataLoader] = None,
                 test_dataloader: Optional[PromptDataLoader] = None, ):
        super().__init__(model=model, config=config, train_dataloader=train_dataloader,
                         valid_dataloader=valid_dataloader, test_dataloader=test_dataloader, )

    def inference_step(self, batch, batch_idx):
        target = batch['tgt_text']  # TODO pop?
        _, pred = self.model.generate(batch, **self.config.generation)
        return pred, target  # these are already a cpu list

    def inference_epoch_end(self, split, outputs):
        preds = []
        targets = []
        for pred, target in outputs:
            preds.extend(pred)
            targets.extend(target)

        self.save_results(split, {
            'preds': preds, 'targets': targets
            })

        metrics = OrderedDict()
        for metric_name in self.config.generation.metric:
            metric = generation_metric(preds, targets, metric_name)
            metrics[metric_name] = metric
        return metrics

    def training_step(self, batch, batch_idx):
        loss = self.model(batch)
        return loss
