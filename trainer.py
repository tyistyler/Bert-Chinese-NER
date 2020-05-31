import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import MODEL_CLASSES, set_seed, compute_metrics, get_intent_labels, get_slot_labels, compute_metrics_slot

logger = logging.getLogger(__name__)

#Trainer(args, train_dataset, dev_dataset, test_dataset)
class Trainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)#获取所有意图标签
        self.slot_label_lst = get_slot_labels(args)#获取所有槽值标签
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index# -100



        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]#BertConfig, JointBERT， BertTokenizer
        self.bert_config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        #here
        self.model = self.model_class(self.bert_config, args, self.intent_label_lst, self.slot_label_lst)#JointBERT(BertConfig, args, intent_label_lst, slot_label_lst)
        # self.model = self.model_class(self.bert_config, args, self.slot_label_lst)  # JointBERT(BertConfig, args, intent_label_lst, slot_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        # 定义采样方式，对象为样本特征
        train_sampler = RandomSampler(self.train_dataset)
        # 构建dataloader，dataloader本质是一个可迭代对象,batch_size=16
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps# 10000
            # 10000/500/2=10
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:# 500/2*10
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)优化器
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()#梯度归0

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")#进度条<--tqdm, 例如分成10组
        # 随机数种子seed确定时，模型的训练结果将始终保持一致
        set_seed(self.args)

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")#进度条, desc为进度条前缀
            for step, batch in enumerate(epoch_iterator):
                self.model.train()#训练模式
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU  把输入数据 放到设备上
                # 分类任务是单句子，不需要设置token_type_ids 默认全为0，batch[2]
                # 定义模型的输入参数 字典形式
                inputs = {'input_ids': batch[0],# 第一行
                          'attention_mask': batch[1],# 第二行
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]


                outputs = self.model(**inputs)#得到模型输出JointBert(input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids, corpus_label_ids)
                loss = outputs[0]#返回值第一个为loss

                # 每个batch都将loss除以gradient_accumulation_steps
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()# 累加损失
                # 每n个batch(把这些batch的梯度求和)，更新一次参数
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    # 反向传播更新参数
                    optimizer.step()
                    # 更新学习率
                    scheduler.step()  # Update learning rate schedule
                    # 清空梯度
                    self.model.zero_grad()
                    global_step += 1

                    # logging_steps = 200
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

    def evaluate(self, mode):#test
        # We use test dataset because semeval doesn't have dev dataset
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()#验证模式

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():# 关闭梯度计算
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4],}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits,) = outputs[:2]# 模型输出的前两项为loss和logits
                # tmp_eval_loss, (slot_logits,), x= outputs[:2]# 模型输出的前两项为loss和logits


                eval_loss += tmp_eval_loss.mean().item()# item()返回一个值
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)


            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps# 平均损失
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)#axis=1：按行查找最大元素 axis=0：按列查找最大元素


        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):# shape[0]-->一行有多少数
            for j in range(out_slot_labels_ids.shape[1]):# shape[1]-->一列有多少数
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])# 真实标签
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])# 预测标签

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        # total_result = compute_metrics_slot(slot_preds_list, out_slot_label_list)

        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self):
        # Save model checkpoint (Overwrite)
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
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.bert_config = self.config_class.from_pretrained(self.args.model_dir)
            logger.info("***** Config loaded *****")
            self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config,
                                                          args=self.args, intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            # self.model = self.model_class.from_pretrained(self.args.model_dir, config=self.bert_config,
            #                                               args=self.args,slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

    def _convert_texts_to_tensors(self, texts, tokenizer,
                                  cls_token_segment_id=0,
                                  pad_token_segment_id=0,
                                  sequence_a_segment_id=0,
                                  mask_padding_with_zero=True):
        """
        Only add input_ids, attention_mask, token_type_ids
        Labels aren't required.
        """
        # Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        input_ids_batch = []
        attention_mask_batch = []
        token_type_ids_batch = []
        slot_label_mask_batch = []

        for text in texts:
            tokens = []
            slot_label_mask = []
            for word in text.split():
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
                # Real label position as 0 for the first token of the word, and padding ids for the remaining tokens
                slot_label_mask.extend([0] + [self.pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > self.args.max_seq_len - special_tokens_count:
                tokens = tokens[:(self.args.max_seq_len - special_tokens_count)]
                slot_label_mask = slot_label_mask[:(self.args.max_seq_len - special_tokens_count)]

            # Add [SEP] token
            tokens += [sep_token]
            slot_label_mask += [self.pad_token_label_id]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # Add [CLS] token
            tokens = [cls_token] + tokens
            slot_label_mask = [self.pad_token_label_id] + slot_label_mask
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = self.args.max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            slot_label_mask = slot_label_mask + ([self.pad_token_label_id] * padding_length)

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            token_type_ids_batch.append(token_type_ids)
            slot_label_mask_batch.append(slot_label_mask)

        # Making tensor that is batch size of 1
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long).to(self.device)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long).to(self.device)
        token_type_ids_batch = torch.tensor(token_type_ids_batch, dtype=torch.long).to(self.device)
        slot_label_mask_batch = torch.tensor(slot_label_mask_batch, dtype=torch.long).to(self.device)

        return input_ids_batch, attention_mask_batch, token_type_ids_batch, slot_label_mask_batch

    def predict(self, texts, tokenizer):
        batch = self._convert_texts_to_tensors(texts, tokenizer)

        slot_label_mask = batch[3]
        self.model.eval()

        # We have only one batch
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'intent_label_ids': None,
                      'slot_labels_ids': None}
            if self.args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[2]
            outputs = self.model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]  # loss doesn't needed

        # Intent prediction
        intent_preds = intent_logits.detach().cpu().numpy()
        intent_preds = np.argmax(intent_preds, axis=1)
        intent_list = []
        for intent_idx in intent_preds:
            intent_list.append(self.intent_label_lst[intent_idx])

        # Slot prediction
        if self.args.use_crf:
            slot_preds = np.array(self.model.crf.decode(slot_logits))
        else:
            slot_preds = slot_logits.detach().cpu().numpy()
            slot_preds = np.argmax(slot_preds, axis=2)

        out_slot_labels_ids = slot_label_mask.detach().cpu().numpy()

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        # Make output.txt with texts, intent_list and slot_preds_list
        with open(os.path.join(self.args.pred_dir, self.args.pred_output_file), 'w', encoding='utf-8') as f:
            for text, intent, slots in zip(texts, intent_list, slot_preds_list):
                f.write("text: {}\n".format(text))
                f.write("intent: {}\n".format(intent))
                f.write("slots: {}\n".format(' '.join(slots)))
                f.write("\n")

        # print output.json
        with open(os.path.join(self.args.pred_dir, self.args.pred_output_file), 'r', encoding='utf-8') as f:
            print(f.read())


