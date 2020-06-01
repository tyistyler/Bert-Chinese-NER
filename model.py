import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, DistilBertModel, AlbertModel
from torchcrf import CRF
import torch.nn.functional as F

PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'distilbert': DistilBertModel,
    'albert': AlbertModel
}

class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class JointBERT(BertPreTrainedModel):
    def __init__(self, bert_config, args, slot_label_lst):
        super(JointBERT, self).__init__(bert_config)#继承父类属性bert_config
        self.args = args
        self.num_slot_labels = len(slot_label_lst)
        # self.lstm = nn.LSTM(input_size=768, hidden_size=128, num_layers=1, bidirectional=True, batch_first=True)

        self.dropout = nn.Dropout(0.3)
        if args.do_pred:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=bert_config)
        else:
            self.bert = PRETRAINED_MODEL_MAP[args.model_type].from_pretrained(args.model_name_or_path, config=bert_config)

        self.slot_classifier = SlotClassifier(bert_config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

        self.slot_pad_token_idx = slot_label_lst.index(args.slot_pad_label)#arg.slot_pad_label=PAD

    def forward(self, input_ids, attention_mask, token_type_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        sequence_output = outputs[0]
        pooled_output = outputs[1]#  [CLS]

        '''
        sequence_output, (hn, cn) = self.lstm(sequence_output)
        '''

        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        if slot_labels_ids is not None:
            if self.args.use_crf:
                '''
                    Make new slot_labels_ids, changing ignore_index(-100) to PAD index in slot label
                    In torchcrf, if index is lower than 0, it makes error when indexing the list
                '''
                padded_slot_labels_ids = slot_labels_ids.detach().clone()
                padded_slot_labels_ids[padded_slot_labels_ids == self.args.ignore_index] = self.slot_pad_token_idx

                slot_loss = self.crf(slot_logits, padded_slot_labels_ids, mask=attention_mask.bype(), reduction='mean')
                slot_loss = -1 * slot_loss# negative log_likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids(-1))

            total_loss += self.args.slot_loss_coef * slot_loss
        outputs = (slot_logits,) + outputs[1:]
        outputs = (total_loss,) + outputs
        return outputs



