#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from mmbt.utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        with numpy_seed(0):
            for row in self.data:
                if np.random.random() < args.drop_img_percent:
                    row["img"] = None

        self.max_seq_len = args.max_seq_len
        if args.model in ["mmbt", "mmbtp", "mmdbt", "mmbt3"]:
            self.max_seq_len -= args.num_image_embeds

        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.args.task == "vsnli":
            sent1 = self.tokenizer(self.data[index]["sentence1"])
            sent2 = self.tokenizer(self.data[index]["sentence2"])
            truncate_seq_pair(sent1, sent2, self.args.max_seq_len - 3)
            sentence = self.text_start_token + sent1 + ["[SEP]"] + sent2 + ["[SEP]"]
            segment = torch.cat(
                [torch.zeros(2 + len(sent1)), torch.ones(len(sent2) + 1)]
            )
        elif self.args.model == "mmbt3":
            sentence = self.tokenizer(self.data[index]["text"])[:(self.max_seq_len - 3)] # -2 for [CLS] and [SEP] tokens
            segment = torch.zeros(len(sentence))
        else:
            sentence = (
                self.text_start_token
                + self.tokenizer(self.data[index]["text"])[:(self.args.max_seq_len - 1)]
            )
            segment = torch.zeros(len(sentence))

        sentence = torch.LongTensor(
            [
                self.vocab.stoi[w] if w in self.vocab.stoi else self.vocab.stoi["[UNK]"]
                for w in sentence
            ]
        )

        if self.args.task_type == "multilabel":
            label = torch.zeros(self.n_classes)
            label[
                [self.args.labels.index(tgt) for tgt in self.data[index]["label"]]
            ] = 1
        else:
            label = torch.LongTensor(
                [self.args.labels.index(self.data[index]["label"])]
            )

        image = None
        if self.args.model in ["img", "concatbow", "concatbow16", "gmu", "concatbert", "mmbt", "mmtr", "mmbtp", "mmdbt", "vilbert", "mmbt3"]:
            '''
            # Extracted vgg16 features
            if self.data[index]["img"]:
                image = torch.load(os.path.join(self.data_dir, 'dataset_img/'+self.data[index]['img'].split('/')[-1].replace('.jpeg', '.pt')))
                seq_len = image.size()[0]
                
                if seq_len > self.args.num_image_embeds:
                    image = image[:self.args.num_image_embeds,]
                elif seq_len < self.args.num_image_embeds:
                    num_missing = self.args.num_image_embeds - seq_len
                    image = torch.cat([image, 128*torch.ones([num_missing,4096])], dim=0)
            else:
                image = 128*torch.ones([self.args.num_image_embeds,4096])
            #image = self.transforms(image)
            '''
            #'''
            # Original
            if self.data[index]["img"]:
                image = Image.open(
                    os.path.join(self.data_dir, self.data[index]["img"])
                ).convert("RGB")
            else:
                image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))
            image = self.transforms(image)
            #'''
            '''
            # Extracted image regions from Faster R-CNN
            if self.data[index]["img"]:
                full_path = os.path.join(self.data_dir, 'dataset_img_raw/'+self.data[index]['img'].split('/')[-1].replace('.jpeg', '.npz'))
                m = np.load(full_path)
                
                regions_list = []
                for arr_name in m.files:
                    region = Image.fromarray(m[arr_name].astype('uint8'), 'RGB')
                    region = self.transforms(region)
                    regions_list.append(region)
                    
                seq_len = len(regions_list)
                if seq_len > self.args.num_images:
                    regions_list = regions_list[:self.args.num_images]
                elif seq_len < self.args.num_images:
                    num_missing = self.args.num_images - seq_len
                    
                    for i in range(num_missing):
                        regions_list.append(128*torch.ones([3,224,224]))
                    
                image = torch.stack(regions_list, dim=0)
            '''
        
        if self.args.model == "mmbt":
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            segment += 1

        return sentence, segment, image, label