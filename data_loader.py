import os
import copy
import logging
import json

import torch
from torch.utils.data import TensorDataset

from utils import get_slot_labels

logger = logging.getLogger(__name__)

class InputExample(object):
    """
        A single training/test example for simple sequence classificatin
        Args:
            guid : Unique id for the example
            words: list. The words of the sequence
            slot_labels list. The slot labels of the example
    """
    def __init__(self, guid, words, slot_labels=None):
        self.guid = guid
        self.words = words
        self.slot_labels = slot_labels

    def __repr__(self):#此时，终端会打印出信息
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n" #indent是缩进打印

class InputFeatures(object):
    """
        A single set of features of data
    """
    def __init__(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class JointPrecessor(object):
    """
        Processor for the JointBERT data set
    """
    def __init__(self, args):
        self.args = args
        self.slot_labels = get_slot_labels(args)

        self.input_text_file = 'seq.in'
        self.slot_labels_file  = 'seq.out'

    @classmethod#不需要实例化
    def _read_file(clscls, input_file, quotechar=None):#读取文件
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = []
            for line in f:
                lines.append(line.strip())
            return lines

    def _create_examples(self, texts, slots, set_type):
        """
            Create examples for the training and dev sets
        """
        examples = []
        for i, (text,slot) in enumerate(zip(texts,slots)):
            guid = "%s-%s" % (set_type, i)
            # input_text
            words = text.split()
            # slot
            slot_labels = []
            for s in slot.split():
                slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)#不满足条件，触发异常
            examples.append(InputExample(guid=guid, words=words, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        '''
        :param mode: train、dev、test
        '''
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.info("looking at {}".format(data_path))
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode)

processors = {
    "weibo":JointPrecessor,
    "resume":JointPrecessor,
    "msra":JointPrecessor
}

#把InputExamples对象，转化为输入特征InputFeatures
def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                                        pad_token_label_id=-100,
                                                        cls_token_segment_id=0,
                                                        pad_token_segment_id=0,
                                                        sequence_a_segment_id=0,
                                                        mask_padding_with_zero=True):
    # Setting based on the current model type.
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing wxample %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token] #for hadling the bad-encoded word
            tokens.extend(word_tokens)
            #Use the real label id for the first token of the word. and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] *(len(word_tokens)-1))
        #Account for [CLS] and [SEP]
        special_tokens_count = 2
        #超出最大长度，则截取前一部分
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        #对于句子对任务，属于句子A的token=0，句子B的token=1；对于分类任务，只有一个输入句子，token全为0
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        #Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        #token在词汇表中的索引
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        #The mask has 1 for real tokens and 0 for paddin tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        #确保长度相同
        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              slot_labels_ids=slot_labels_ids))
    return features

def load_and_cache_examples(args, tokenizer, mode):#mode=train/dev/test
    processor = processors[args.task](args)

    #Load data features form cache or datase file，特征保存目录的命名
    cached_features_file = os.path.join(
            args.data_dir,
            'cached_{}_{}_{}_{}'.format(
                mode, args.task,
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                args.max_seq_len
            )
    )
    #如果已经构造好了特征，则直接加载，避免重复处理
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached filt %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        #对验证集、测试集、训练集进行处理,把数据转换为InputExample对象
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "dev":
            examples = processor.get_examples("dev")
        elif  mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train,dev,test is available")
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer, pad_token_label_id)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    #Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids,
                            all_slot_labels_ids)
    return dataset
