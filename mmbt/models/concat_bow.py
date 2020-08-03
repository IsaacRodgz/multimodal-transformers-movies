#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from mmbt.models.bow import GloveBowEncoder
from mmbt.models.image import ImageEncoder
from mmbt.models.image import ImageEncoder16


class MaxOut(nn.Module):
    def __init__(self, input_dim, output_dim, num_units=2):
        super(MaxOut, self).__init__()
        self.fc1_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(num_units)])

    def forward(self, x): 

        return self.maxout(x, self.fc1_list)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class MultimodalConcatBowClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBowClf, self).__init__()
        self.args = args
        self.clf = nn.Linear(
            args.embed_sz + (args.img_hidden_sz * args.num_image_embeds), args.n_classes
        )
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder(args)

    def forward(self, txt, img):
        txt = self.txtenc(txt)
        img = self.imgenc(img)
        img = torch.flatten(img, start_dim=1)
        cat = torch.cat([txt, img], -1)
        return self.clf(cat)


class MLPGenreClf(nn.Module):

    def __init__(self, args):

        super(MLPGenreClf, self).__init__()
        self.args = args
        
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder16(args)

        self.bn1 = nn.BatchNorm1d(args.embed_sz+args.img_hidden_sz)
        self.linear1 = MaxOut(args.embed_sz+args.img_hidden_sz, args.hidden_sz)
        self.drop1 = nn.Dropout(p=args.dropout)
        
        self.bn2 = nn.BatchNorm1d(args.hidden_sz)
        self.linear2 = MaxOut(args.hidden_sz, args.hidden_sz)
        self.drop2 = nn.Dropout(p=args.dropout)
        
        self.bn3 = nn.BatchNorm1d(args.hidden_sz)
        self.linear3 = nn.Linear(args.hidden_sz, args.n_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, txt, img):
        txt = self.txtenc(txt)
        img = self.imgenc(img)
        x = torch.cat([txt, img], -1)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.bn2(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.bn3(x)
        x = self.linear3(x)

        return x


class MultimodalConcatBow16Clf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBow16Clf, self).__init__()
        self.args = args
        self.clf = nn.Linear(
            args.embed_sz + args.img_hidden_sz, args.n_classes
        )
        
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder16(args)

    def forward(self, txt, img):
        txt = self.txtenc(txt)
        img = self.imgenc(img)
        cat = torch.cat([txt, img], -1)
        return self.clf(cat)