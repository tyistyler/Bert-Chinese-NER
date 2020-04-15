import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, DistilBertModel, AlbertModel, DistilBertPreTrainedModel
# from torchcrf import CRF

PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'albert': AlbertModel
}

num_word = 10

class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)#抑制过拟合，随机删除一些神经元
        self.linear = nn.Linear(input_dim, num_intent_labels)#最后输出的维度必须是意图标签的个数

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)



class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)






class JointBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args, intent_label_lst, slot_label_lst):#config配置
    # def __init__(self, bert_config, args, slot_label_lst):  # config配置
        super(JointBERT, self).__init__(bert_config)#继承父类属性bert_config
        self.args = args
        self.num_intent_labels = len(intent_label_lst)#intenet_label_lst获取所有意图标签
        self.num_slot_labels = len(slot_label_lst)#获取所有槽值标签
        if args.do_pred:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=bert_config)#BertModel(config=bert_config)
        else:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(bert_config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(bert_config.hidden_size, self.num_slot_labels, args.dropout_rate)

        # if args.use_crf:
            # self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.slot_pad_token_idx = slot_label_lst.index(args.slot_pad_label)#arg.slot_pad_label=PAD


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0] #       dim=3
        pooled_output = outputs[1]  # [CLS]  dim=2


        # *******************************************************************************************************************************************

        # ********************************************************************************************************************************************

        intent_logits = self.intent_classifier(pooled_output)#线性映射
        slot_logits = self.slot_classifier(sequence_output)#线性映射

        total_loss = 0

        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()#input, target
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))#view(-1)，将矩阵转化为1维，num_intent_labels共有多个意图标签
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                # Make new slot_labels_ids, changing ignore_index(-100) to PAD index in slot label
                # In torchcrf, if index is lower than 0, it makes error when indexing the list
                padded_slot_labels_ids = slot_labels_ids.detach().clone()
                padded_slot_labels_ids[padded_slot_labels_ids == self.args.ignore_index] = self.slot_pad_token_idx#0

                slot_loss = self.crf(slot_logits, padded_slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[1:]  # add hidden states and attention if they are here
        # outputs = ((slot_logits),) + outputs[1:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


class JointDistilBERT(DistilBertPreTrainedModel):
    def __init__(self, distilbert_config, args, intent_label_lst, slot_label_lst):
        super(JointDistilBERT, self).__init__(distilbert_config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        if args.do_pred:
            self.distilbert = PRETRAINED_MODEL_MAP[args.model_type](config=distilbert_config)
        else:
            self.distilbert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path,
                                                                                    config=distilbert_config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(distilbert_config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(distilbert_config.hidden_size, self.num_slot_labels, args.dropout_rate)

        # if args.use_crf:
            # self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.slot_pad_token_idx = slot_label_lst.index(args.slot_pad_label)

    def forward(self, input_ids, attention_mask, intent_label_ids, slot_labels_ids):
        outputs = self.distilbert(input_ids, attention_mask=attention_mask)  # last-layer hidden-state, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                # Make new slot_labels_ids, changing ignore_index(-100) to PAD index in slot label
                # In torchcrf, if index is lower than 0, it makes error when indexing the list
                padded_slot_labels_ids = slot_labels_ids.detach().clone()
                padded_slot_labels_ids[padded_slot_labels_ids == self.args.ignore_index] = self.slot_pad_token_idx

                slot_loss = self.crf(slot_logits, padded_slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[1:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits





