import argparse
import os

import torch
from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, MODEL_CLASSES, MODEL_PATH_MAP
from data_loader import load_and_cache_examples, load_and_cache_examples_snips
from transformers import BertConfig, BertModel, BertTokenizer
from utils import get_snips_tensors
import logging
from torch.utils.data import DataLoader


logger = logging.getLogger(__name__)
def main(args):
# def main():
    init_logger()#输出信息
    tokenizer = load_tokenizer(args)#加载预训练模型

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")#获取样本特征
    dev_dataset   = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset  = load_and_cache_examples(args, tokenizer, mode="test")

    # snips_tensors表示一个二维矩阵，代表了snips数据集中每句话对应的CLS输出
    '''
        数据格式 TensorDataset(all_corpus, all_input_ids, all_attention_mask,
                               all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
                            
    '''


    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.load_model()
        trainer.evaluate("test")

    if args.do_pred:
        trainer.load_model()
        texts = read_prediction_text(args)
        trainer.predict(texts, tokenizer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, required=True, type=str, help="The name of the task to train")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--data_dir", default="./data", type=str, help="The input data dir")
    parser.add_argument("--intent_label_file", default="intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="slot_label.txt", type=str, help="Slot Label file")

    # parser.add_argument("--corpus_label_file", default="corpus_label.txt", type=str, help="Corpus Label file")

    parser.add_argument("--model_type", default="bert", type=str, help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for training and evaluation.")#batch_size=16
    parser.add_argument("--max_seq_len", default=250, type=int, help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")

    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--ignore_index", default=-100, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # For prediction
    parser.add_argument("--pred_dir", default="./preds", type=str, help="The input prediction dir")
    parser.add_argument("--pred_input_file", default="preds.txt", type=str, help="The input text file of lines for prediction")
    parser.add_argument("--pred_output_file", default="outputs.txt", type=str, help="The output file of prediction")
    parser.add_argument("--do_pred", action="store_true", help="Whether to predict the sentences")

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    # 解析参数
    args = parser.parse_args()#将之前add_argument()定义的参数进行赋值，并返回相关的namespace，即解析参数

    args.model_name_or_path = MODEL_PATH_MAP[args.model_type]#bert-base-uncased
    # print(args.model_name_or_path)
    main(args)
