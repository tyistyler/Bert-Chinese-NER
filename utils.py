import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import BertPreTrainedModel, BertTokenizer, BertConfig, DistilBertConfig, DistilBertTokenizer, AlbertConfig, \
                                                    AlbertTokenizer, BertModel, DistilBertModel, AlbertModel

from model import JointBERT, JointDistilBERT

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distilbert': (DistilBertConfig, JointDistilBERT, DistilBertTokenizer),
    'albert': (AlbertConfig, JointBERT, AlbertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-chinese',#中文
    'distilbert': 'distilbert-base-uncased',
    'albert': 'albert-xxlarge-v1'
}
PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'albert': AlbertModel
}



def get_intent_labels(args):#获取所有意图标签
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]

def get_corpus_labels(args):#获取所有语料库标签
    return [label.strip() for label in open(os.path.join(args.data_dir,args.task, args.corpus_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):#获取所有槽值标签
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

# BertTokenizer.from_pretrained(bert-base-uncased)
def load_tokenizer(args):
    return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)




def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


# 为当前GPU设置随机种子，以使得结果是确定的
# 不加入manual_seed时，随机数会变化
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)#为CPU设置种子
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)#为GPU设置种子


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results
def compute_metrics_slot(slot_preds, slot_labels):
    assert len(slot_preds) == len(slot_labels)
    results = {}
    slot_result = get_slot_metrics(slot_preds, slot_labels)

    results.update(slot_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    # for i in range(preds.size):#输出正确句子的编号
    #     if (preds[i] == labels[i]):
    #         with open('data/seq.txt', 'a', encoding='utf-8') as f:
    #             f.write(''.join(str(i)) + '\n')
    # f.close()

    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }
# def get_corpus_acc(preds, labels):
#     acc = (preds == labels).mean()
#     return{
#         "corpus_acc":acc
#     }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]



def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }



class get_snips_tensors(BertPreTrainedModel):
    def __init__(self, bert_config, args, snips_train_dataset):
        super(get_snips_tensors, self).__init__(bert_config)
        self.args = args
        self.snips_train_dataset = snips_train_dataset

        self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)

    def forward(self, input_ids, attention_mask, token_type_ids):

        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        pooled_output = output[1]# CLS对应输出

        return pooled_output
