#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import functools
import json
import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from pytorch_pretrained_bert import BertTokenizer
from transformers import DistilBertTokenizer
from torch.utils.data import DataLoader

from mmbt.data.dataset import JsonlDataset
from mmbt.data.vocab import Vocab


def get_transforms(args):
    '''
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    '''
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.46777044, 0.44531429, 0.40661017],
                std=[0.12221994, 0.12145835, 0.14380469],
            ),
        ]
    )


def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]

    if type(data_labels[0]) == list:
        for label_row in data_labels:
            label_freqs.update(label_row)
    else:
        label_freqs.update(data_labels)

    return list(label_freqs.keys()), label_freqs


def get_glove_words(path):
    word_list = []
    for line in open(path):
        w, _ = line.split(" ", 1)
        word_list.append(w)
    return word_list


def get_vocab(args):
    vocab = Vocab()
    if args.model in ["bert", "mmbt", "concatbert", "mmtr", "mmtrv", "mmtra", "mmtrva", "mmtrta", "mmtrvap", "mmtrvapt", "mmtrvpp", "mmtrvppm", "mmtrvpapm", "mmtrvpa", "mmbtp", "vilbert", "mmbt3", "mmvilbt", "mmbtratingtext", "mmbtadapter", "mmbtadapterm"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
        
    elif args.model in ["mmdbt", "mmbtrating", "mmtrrating"]:
        distilbert_tokenizer = DistilBertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        
        vocab.stoi = distilbert_tokenizer.vocab
        vocab.itos = distilbert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)

    else:
        word_list = get_glove_words(args.glove_path)
        vocab.add(word_list)

    return vocab


def collate_fn(batch, args):
    
    bsz = len(batch)
    
    text_tensor = segment_tensor = mask_tensor = None
    if args.model not in ["mmtra", "mmtrva", "mmtrvap"]:
        lens = [len(row[0]) for row in batch]
        max_seq_len = max(lens)

        if args.model == "mmbt3":
            mm_max_seq_len = args.max_seq_len - (args.num_image_embeds+2) - (max_seq_len+1)
            mm_seq_len = min(mm_max_seq_len, max_seq_len+args.num_image_embeds)
            mm_seq_len = max(0, mm_seq_len)
            mask_tensor = torch.zeros(bsz, max_seq_len+1).long()
            mm_mask_tensor = torch.zeros(bsz, mm_seq_len).long()
        else:
            mask_tensor = torch.zeros(bsz, max_seq_len).long()

        text_tensor = torch.zeros(bsz, max_seq_len).long()
        segment_tensor = torch.zeros(bsz, max_seq_len).long()

    img_tensor = None
    if args.model in ["img", "concatbow", "concatbow16", "gmu", "concatbert", "mmbt", "mmtr", "mmtrv", "mmtrva", "mmtrvap", "mmtrvapt", "mmtrvpp", "mmtrvpa", "mmbtp", "mmdbt", "vilbert", "mmbt3", "mmvilbt", "mmbtrating", "mmtrrating", "mmbtadapter"] or args.visual in ["video", "both"]:
        img_tensor = torch.stack([row[2] for row in batch])
        
    genres = None
    if args.task == "mpaa":
        genres = torch.stack([row[4] for row in batch])
        
    poster = None
    audio = None
    metadata = None
    if args.task == "moviescope":
        if args.model in ["mmtrta"]:
            img_lens = [row[2].shape[1] for row in batch]
            img_min_len = min(img_lens)
            audio = torch.stack([row[2][..., :img_min_len] for row in batch])
        elif args.model in ["mmtra", "mmtrva", "mmtrvap", "mmtrvapt", "mmtrvpa", "mmtrvpapm", "mmbt"]:
            img_lens = [row[4].shape[1] for row in batch]
            img_min_len = min(img_lens)
            audio = torch.stack([row[4][..., :img_min_len] for row in batch])
        if args.visual in ["poster", "both"]:
            if args.model in ["mmtrvap", "mmtrvapt", "mmtrvpapm", "mmbt"]:
                poster = torch.stack([row[5] for row in batch])
            elif args.model in ["bert"]:
                pass
            else:
                poster = torch.stack([row[4] for row in batch])
        if args.model in ["mmtrvppm", "mmtrvpapm", "mmbt"]:
            if args.model in ["mmtrvpapm", "mmbt"]:
                metadata = torch.stack([row[6] for row in batch])
            else:
                metadata = torch.stack([row[5] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        if args.model in ["bert"]:
            tgt_tensor = torch.stack([row[2] for row in batch])
        else:
            tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        if args.model in ["bert"]:
            tgt_tensor = torch.cat([row[2] for row in batch]).long()
        else:
            tgt_tensor = torch.cat([row[3] for row in batch]).long()
    
    if args.model not in ["mmtra", "mmtrva", "mmtrvap"]:
        for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
            tokens, segment = input_row[:2]
            text_tensor[i_batch, :length] = tokens
            segment_tensor[i_batch, :length] = segment
            if args.model == "mmbt3":
                mask_tensor[i_batch, :length+1] = 1
                mm_mask_tensor[i_batch, :length+args.num_image_embeds] = 1
            else:
                mask_tensor[i_batch, :length] = 1
    
    if args.model == "mmbt3":
        return text_tensor, segment_tensor, mask_tensor, mm_mask_tensor, img_tensor, tgt_tensor, genres
    elif args.task == "moviescope":
        if args.model in ["mmtrta"]:
            return text_tensor, segment_tensor, mask_tensor, audio, tgt_tensor
        elif args.model in ["mmtra", "mmtrva", "mmtrvpa"]:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, audio
        elif args.model == "mmtrvppm":
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, poster, metadata
        elif args.model in ["mmtrvpapm", "mmbt"]:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, audio, poster, metadata
        elif args.model in ["mmtrvap", "mmtrvapt"]:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, audio, poster
        elif args.model in ["bert"]:
            return text_tensor, segment_tensor, mask_tensor, tgt_tensor
        else:
            return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, poster
    else:
        return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor, genres


def get_data_loaders(args, data_all=None, partition_index=None):
    '''
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert", "mmtr", "mmbtp"]
        else str.split
    )
    '''
    if args.model in ["bert", "mmbt", "concatbert", "mmtr", "mmtrv", "mmtra", "mmtrva", "mmtrta", "mmtrvap", "mmtrvapt", "mmtrvpp", "mmtrvppm", "mmtrvpapm", "mmtrvpa", "mmbtp", "vilbert", "mmbt3", "mmvilbt", "mmbtratingtext", "mmbtadapter", "mmbtadapterm"]:
        tokenizer = (
            BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )
    
    elif args.model in ["mmdbt", "mmbtrating", "mmtrrating"]:
        tokenizer = (
            DistilBertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )
    
    else:
        tokenizer = (str.split)

    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    if args.task == "mpaa":
        genres = [g for line in open(os.path.join(args.data_path, args.task, "train.jsonl")) for g in json.loads(line)["genre"]]
        args.genres = list(set(genres))
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)
    
    if args.train_type == "split":

        train = JsonlDataset(
            os.path.join(args.data_path, args.task, "train.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        args.train_data_len = len(train)

        dev = JsonlDataset(
            os.path.join(args.data_path, args.task, "dev.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        collate = functools.partial(collate_fn, args=args)

        train_loader = DataLoader(
            train,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers,
            collate_fn=collate,
            drop_last=True,
        )

        val_loader = DataLoader(
            dev,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        test_set = JsonlDataset(
            os.path.join(args.data_path, args.task, "test.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        return train_loader, val_loader, test_loader
    
    else:
        dev_size = int(len(data_all)*0.2)
        train_size = len(data_all)-dev_size
        k = partition_index
        dev_start = k*dev_size
        dev_end = (k+1)*dev_size
        
        if k == 0:
            train_data = data_all[dev_end:]
        elif k == 9:
            train_data = data_all[:dev_start]
        else:
            train_data = data_all[:dev_start] + data_all[dev_end:]
        dev_data = data_all[dev_start:dev_end]
        
        test_size = int(len(train_data)*0.1)
        
        train = JsonlDataset(
            os.path.join(args.data_path, args.task, "train.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
            train_data[test_size:],
        )

        args.train_data_len = len(train)

        dev = JsonlDataset(
            os.path.join(args.data_path, args.task, "dev.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
            dev_data,
        )

        collate = functools.partial(collate_fn, args=args)

        train_loader = DataLoader(
            train,
            batch_size=args.batch_sz,
            shuffle=True,
            num_workers=args.n_workers,
            collate_fn=collate,
        )

        val_loader = DataLoader(
            dev,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )
        
        test_set = JsonlDataset(
            os.path.join(args.data_path, args.task, "test.jsonl"),
            tokenizer,
            transforms,
            vocab,
            args,
            train_data[:test_size],
        )

        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_sz,
            shuffle=False,
            num_workers=args.n_workers,
            collate_fn=collate,
        )
        
        return train_loader, test_loader, val_loader