import os
import logging
from tqdm import tqdm,trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, set_seed, compute_metrics, get_slot_labels

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.slot_label_lst = get_slot_labels(args)
        #Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index#-100

        self.config_class, self.model_class, _ =MODEL_CLASSES[args.model_type]#BertConfig, JointBERT, BertTokenizer
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)

        self.model = self.model_class(self.bert_config, args, self.slot_label_lst)

        #Gpu of CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        #定义采样方式，对象为样本特征
        train_sampler = RandomSampler(self.train_dataset)
        #构建dataloader，其本质是一个可迭代对象，batch_size=16
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params':[p for n,p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': self.args.weight_decay},
            {'params':[p for n,p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("   Num wxamples = %d", len(self.train_dataset))
        logger.info("   Num epochs = %d", self.args.num_train_epochs)
        logger.info("   Total train batch size = %d", self.args.batch_size)
        logger.info("   Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("   Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")#进度条，如分成10组
        set_seed(self.args)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                            'input_ids':batch[0],
                            'attention_mask':batch[1],
                            'slot_labels_ids':batch[3]
                }
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]

                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                # 每n个batch(把这些batch的梯度求和)，更新一次参数
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    #梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    #反 向传播，更新参数
                    optimizer.step()
                    # 更新学习率
                    scheduler.step()
                    # 清空梯度
                    self.model.zero_grad()
                    global_step += 1

                    #logging_steps = 200
                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate("dev")# fine-tuning

                    # save_steps = 200
                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break
        return global_step, tr_loss / global_step

    def evaluate(self, mode):

        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)
        #Eval!
        # logger.info("*** Running evaluation on %s dataset ******", mode)
        logger.info("   Num examples = %d",len(dataset))
        logger.info("   Batch size = %d", len(self.args.batch_size))
        eval_loss = 0.0
        nb_eval_steps = 0
        slot_preds = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():#不计算梯度
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'slot_labels_ids': batch[3]}
                if self.args.model_type != 'distibert':
                    inputs['token_type_ids'] = batch[2]

                outputs = self.model(**inputs)
                tmp_eval_loss, slot_logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in torchcrf returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()

            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.append(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps#平均损失
        results = {
            "loss": eval_loss
        }
        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):#shape[0]-->行数
            for j in range(out_slot_labels_ids.shape[1]):#shape[1]-->列数
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])# real label
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])# pred label

        total_result = compute_metrics(slot_preds_list, out_slot_label_list)

        results.update(total_result)

        logger.info("*****   Eval results  *****")
        for key in sorted(results.keys()):
            logger.info("   %s = %s",key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overweite)
        output_dir = os.path.join(self.args.model_dir)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, 'training_config.bin'))
        logger.info("Saving model checkpoint to %s", output_dir)
    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exiss! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("*****  Config  loaded  *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config,
                                                          args=self.args, slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("*****  Model  loaded  *****")
        except:
            raise Exception("Some model files might be missing...")







