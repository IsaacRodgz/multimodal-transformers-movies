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
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import pickle

import torch
from torch.utils.data import Dataset

from mmbt.utils.utils import truncate_seq_pair, numpy_seed


class JsonlDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms_, vocab, args, data_dict=None):
        if data_dict is not None:
            self.data = data_dict
        else:
            self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        self.n_classes = len(args.labels)
        self.text_start_token = ["[CLS]"] if args.model != "mmbt" else ["[SEP]"]

        self.max_seq_len = args.max_seq_len
        if args.model in ["mmbt", "mmbtp", "mmdbt", "mmbt3"]:
            #self.max_seq_len -= args.num_image_embeds
            self.max_seq_len -= 3
        
        if self.args.model in ["mmtrvppm", "mmtrvpapm", "mmbt"]:
            split = data_path.split('/')[-1].split('.')[0].replace("dev", "val")
            #split = split if split != 'dev' else 'val'
            metadata_dir = os.path.join(self.data_dir, 'Metadata_matrices', f'{split}_metadata.npy')
            self.metadata_matrix = np.load(metadata_dir)
            metadata_dir = os.path.join(self.data_dir, 'Metadata_matrices', f'{split}_ids.pickle')
            with open(metadata_dir, 'rb') as handle:
                self.metadata_dict = pickle.load(handle)

        self.transforms = transforms_
        self.vilbert_transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.406, 0.456, 0.485],
                    std=[1., 1., 1.],
                ),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = segment = None
        if self.args.model not in ["mmtrva", "mmtrvap"]:
            if self.args.task == "mpaa":
                sentence = self.tokenizer(self.data[index]["script"])
                segment = torch.zeros(len(sentence))
            elif self.args.task == "moviescope":
                sentence = (
                    self.text_start_token
                    + self.tokenizer(self.data[index]["synopsis"])[:(self.max_seq_len - 1)]
                )
                segment = torch.zeros(len(sentence))
            elif self.args.task == "vsnli":
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
        if self.args.model in ["img", "concatbow", "concatbow16", "gmu", "concatbert", "mmbt", "mmtr", "mmtrv", "mmtrva", "mmtrvap", "mmtrvapt", "mmtrvpp", "mmtrvppm", "mmtrvpapm", "mmtrvpa", "mmbtp", "mmdbt", "vilbert", "mmbt3", "mmvilbt", "mmbtrating", "mmtrrating", "mmbtratingtext", "mmbtadapter", "mmbtadapterm"]:
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
            if self.args.task == "moviescope":
                if self.args.visual in ["video", "both"]:
                    file = open(os.path.join(self.data_dir, '200F_VGG16', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    image = torch.from_numpy(data).squeeze(0)
                
                poster = None
                if self.args.visual in ["poster", "both"]:
                    '''
                    image_dir = os.path.join(self.data_dir, 'Raw_Poster', f'{str(self.data[index]["id"])}.jpg')
                    poster = image = Image.open(image_dir).convert("RGB")
                    poster = self.transforms(poster)
                    '''
                    file = open(os.path.join(self.data_dir, 'PosterFeatures', f'{str(self.data[index]["id"])}.p'), 'rb')
                    data = pickle.load(file, encoding='bytes')
                    poster = torch.from_numpy(data).squeeze(0)
            else:
                #'''
                # Original
                if self.data[index]["img"]:
                    if self.args.task == "handwritten":
                        self.data[index]["img"] = "HWxPI-Track-ICPR2018/"+"/".join(self.data[index]["img"].split("/")[1:])

                    image = Image.open(
                        os.path.join(self.data_dir, self.data[index]["img"])
                    ).convert("RGB")
                else:
                    image = Image.fromarray(128 * np.ones((256, 256, 3), dtype=np.uint8))

                if self.args.model != "vilbert":
                    image = self.transforms(image)
                else:
                    image = self.vilbert_transforms(image)
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
                
        audio = None
        if self.args.model in ["mmtra", "mmtrva", "mmtrta", "mmtrvap", "mmtrvapt", "mmtrvpa", "mmtrvpapm", "mmbt"]:
            if self.args.orig_d_a == 96:
                file = open(os.path.join(self.data_dir, 'Melspectrogram', f'{str(self.data[index]["id"])}.p'), 'rb')
                data = pickle.load(file, encoding='bytes')
                audio = torch.from_numpy(data).type(torch.FloatTensor)
            else: 
                file = open(os.path.join(self.data_dir, 'MelgramPorcessed', f'{str(self.data[index]["id"])}.p'), 'rb')
                data = pickle.load(file, encoding='bytes')
                data = torch.from_numpy(data).type(torch.FloatTensor).squeeze(0)
                audio = torch.cat([frame for frame in data[:4]], dim=1)
        
        metadata = None
        if self.args.model in ["mmtrvppm", "mmtrvpapm", "mmbt"]:
            example_id = self.data[index]["id"]
            metadata_idx = self.metadata_dict[example_id]
            metadata = self.metadata_matrix[metadata_idx]
            # metadata = np.delete(metadata, 10, 0) # for rating classification ( remove rating information)
            metadata = torch.from_numpy(metadata).type(torch.FloatTensor)
                    
        if self.args.task == "mpaa":
            genres = torch.zeros(len(self.args.genres))
            genres[[self.args.genres.index(tgt) for tgt in self.data[index]["genre"]]] = 1
                            
        if self.args.model in ["mmbt", "mmbtadapter"]:
            # The first SEP is part of Image Token.
            segment = segment[1:]
            sentence = sentence[1:]
            # The first segment (0) is of images.
            #segment += 1
        
        if self.args.task == "mpaa":
            return sentence, segment, image, label, genres
        elif self.args.task == "moviescope":
            if self.args.model in ["mmtrta"]:
                return sentence, segment, audio, label
            if self.args.model in ["mmtra", "mmtrva", "mmtrvpa"]:
                return sentence, segment, image, label, audio
            elif self.args.model == "mmtrvppm":
                return sentence, segment, image, label, poster, metadata
            elif self.args.model in ["mmtrvpapm", "mmbt"]:
                return sentence, segment, image, label, audio, poster, metadata
            elif self.args.model in ["mmtrvap", "mmtrvapt"]:
                return sentence, segment, image, label, audio, poster
            elif self.args.model in ["bert", "bow"]:
                return sentence, segment, label
            else:
                return sentence, segment, image, label, poster
        else:
            return sentence, segment, image, label