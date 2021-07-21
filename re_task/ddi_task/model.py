import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, RobertaModel, AlbertModel
import torch.nn.functional as F
import numpy as np
import math


PRETRAINED_MODEL_MAP = {
    'bert': BertModel,
    'roberta': RobertaModel,
    'albert': AlbertModel
}


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0., use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Bert_for_re(BertPreTrainedModel):
    def __init__(self, config, args):
        super(Bert_for_re, self).__init__(config)

        self.args = args
        self.num_labels = config.num_labels
        self.bert = PRETRAINED_MODEL_MAP[args.model_type](config=config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)

        #only use the BERT model
        if self.args.model == "only_bert":
            self.label_classifier = FCLayer(config.hidden_size, config.num_labels, args.dropout_rate, use_activation=False)

        #use  BERT + center vector
        if self.args.model == "bert_center":
            self.label_classifier = FCLayer(config.hidden_size * 2, config.num_labels, args.dropout_rate, use_activation=False)

    @staticmethod
    # Averaging treatment
    def average(hidden_output, list):
        list_unsqueeze = list.unsqueeze(1)
        length_tensor = (list != 0).sum(dim=1).unsqueeze(1)
        sum_vector = torch.bmm(list_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, center_list
                ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        pooled_output = self.fc_layer(pooled_output)

        # only BERT model used
        if self.args.model == "only_bert":
            logits = self.label_classifier(pooled_output)
            outputs = (logits,) + outputs[2:]

        # use BERT model and center vector
        if self.args.model == 'bert_center':
            center = self.average(sequence_output, center_list)
            center = self.fc_layer(center)
            concat = torch.cat([center, pooled_output, ], dim=-1)
            logits = self.label_classifier(concat)
            outputs = (logits,) + outputs[2:]


        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)






