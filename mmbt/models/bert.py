#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn
from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel as huggingBertModel


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = huggingBertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        _, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
        )
        return out


class BertClf(nn.Module):
    def __init__(self, args):
        super(BertClf, self).__init__()
        self.args = args
        self.enc = BertEncoder(args)
        #self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        #'''
        self.clf = nn.Sequential(
            nn.Linear(args.hidden_sz, args.hidden_sz),
            nn.BatchNorm1d(args.hidden_sz),
            #nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.hidden_sz, args.n_classes)
        )
        #'''

    def forward(self, txt, mask, segment):
        x = self.enc(txt, mask, segment)
        return self.clf(x)
