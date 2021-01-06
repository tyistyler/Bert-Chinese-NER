import argparse
import os
import torch

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, MODEL_CLASSES, MODEL_PATH_MAP
import logging
from data_loader import load_and_cache_examples

import jieba
import math

from flask import Flask, request
import json


logger = logging.getLogger(__name__)
app = Flask(__name__)

def wordinfile(word, texts):
    count = 0
    for text in texts:
        text = "".join(text)
        if word in text:
            count += 1
    return count

def get_tf_idf(texts, slot_preds_lists, jieba_word_dict):
    jieba_word_dict = jieba_word_dict
    term_word_dict = {}
    file_len = len(texts)       # 文档总数
    for text, slots in zip(texts, slot_preds_lists):
        assert len(text) == len(slots), print(text + '\n' + slots)
        i = 0
        while i < len(slots):
            if slots[i] != 'O' and slots[i] != 'PAD':
                j = i
                while slots[j] != 'O':
                    # print(j)
                    j += 1
                    if j == len(slots):
                        break

                word = ''.join(text[i:j])
                if word is None:
                    continue
                if word not in term_word_dict and word not in jieba_word_dict:
                    term_word_dict[word] = 1
                elif word in term_word_dict and word not in jieba_word_dict:
                    term_word_dict[word] += 1
                i = j
            else:
                i += 1
    # print(len(jieba_word_dict))
    # print(len(new_word_dict))
    all_words_len = len(jieba_word_dict) + len(term_word_dict)
    term_word_tf_idf = {}
    jieba_word_tf_idf = {}
    # 术语tf
    for word in term_word_dict:
        patent_tf = term_word_dict[word] / all_words_len #某个词在文章中出现的次数/文章总词数
        patent_idf = math.log(file_len/(wordinfile(word, texts)+1)) #log(语料库的文档总数)/(包含该词的文档数+1)
        patent_tfidf = patent_tf*patent_idf
        term_word_tf_idf[word] = patent_tfidf
    for word in jieba_word_dict:
        word_tf = jieba_word_dict[word] / all_words_len
        word_idf = math.log(file_len / (wordinfile(word, texts) + 1))  # log(语料库的文档总数)/(包含该词的文档数+1)
        word_tfidf = word_tf * word_idf
        jieba_word_tf_idf[word] = word_tfidf
    # print(term_word_tf_idf)
    # print(jieba_word_tf_idf)
    term_weight = term_word_tf_idf
    for term in term_word_tf_idf:
        for word in jieba_word_tf_idf:
            if word in term:
                term_weight[term] += jieba_word_tf_idf[word]

    term_weight = sorted(term_weight.items(), key=lambda item:item[1], reverse=True)
    # print(term_weight)
    return term_weight
            

def main(args):
    init_logger()#输出信息
    tokenizer = load_tokenizer(args)# 加载预训练模型

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset   = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset  = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    # if args.do_train:
        # trainer.train()
    # if args.do_eval:
        # trainer.load_model()
        # trainer.evaluate("test")
        
        
    @app.route('/pred_term', methods=['GET', 'POST'])
    def get_data():
        if request.method == 'POST':
            argsJson = request.data.decode('utf-8')
            argsJson = json.loads(argsJson)
            (title, texts), = argsJson.items()
            # 结巴分词
            jieba_text = " ".join(jieba.cut(texts, cut_all=False))
            jieba_text = jieba_text.split()
            jieba_word_dict = {}
            for i in jieba_text:
                if i not in jieba_word_dict:
                    jieba_word_dict[i] = 1
                else:
                    jieba_word_dict[i] += 1
            
            # 术语识别
            texts = " ".join(texts)
            texts = texts.split('。')
            if len(texts[-1])==0:
                texts = texts[:-1]
            slot_preds_list = trainer.predict(texts, tokenizer)
            new_texts = []
            for t in texts:
                new_texts.append(t.strip().split())
            # print(new_texts)
            term_weight = get_tf_idf(new_texts, slot_preds_list, jieba_word_dict)
            term_weight = json.dumps(term_weight, ensure_ascii=False)
            return term_weight
        else:
            return " 'it's not a POST operation! "
        
        
        
        
    if args.do_pred:
        trainer.load_model()
        # texts = read_prediction_text(args)
        app.run(host='0.0.0.0', port=5001)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data/BIOES", type=str, help="The input data dir")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot label file")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--seed", type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation")
    parser.add_argument("--max_seq_len", default=500, type=int, help="The maximum total input sequence length after tokenization")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--num_train_epochs", default=30.0, type=float, help="Total number of training epochs to perform")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay if we apply some")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set toal number of training strps to perform.Override num_train_epochs")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument("--logging_steps", type=int, default=200, help="Log every X updates steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every X updates steps")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=-100, type=int,
                        help="Specifies a target value that is ignored and does not contribute to the input gradienet")

    parser.add_argument("--slot_loss_coef", type=float, default=1.0, help="Coefficient for the slot loss")

    #For prediction
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The input prediction dir")
    parser.add_argument("--pred_input_file", default="preds.txt", type=str, help="Teh input text file of lines for prediction")
    parser.add_argument("--pred_output_file", default="outputs.txt", type=str, help="The output file of prediction")
    parser.add_argument("--do_pred", action='store_true', help="Whether to predict the sentences")

    #CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad(to be ignore when cauculate loss")

    #解析参数
    args = parser.parse_args()#将之前定义的参数进行复制，并返回相关的namespace

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]#bert-base-chinese

    main(args)














