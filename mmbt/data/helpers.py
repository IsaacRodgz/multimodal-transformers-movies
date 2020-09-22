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
    if args.model in ["bert", "mmbt", "concatbert", "mmtr", "mmbtp", "vilbert", "mmbt3", "mmvilbt"]:
        bert_tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=True
        )
        vocab.stoi = bert_tokenizer.vocab
        vocab.itos = bert_tokenizer.ids_to_tokens
        vocab.vocab_sz = len(vocab.itos)
        
    elif args.model in ["mmdbt", "mmbtrating"]:
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

    lens = [len(row[0]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

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

    if args.model in ["img", "concatbow", "concatbow16", "gmu", "concatbert", "mmbt", "mmtr", "mmbtp", "mmdbt", "vilbert", "mmbt3", "mmvilbt", "mmbtrating"]:
        img_tensor = torch.stack([row[2] for row in batch])

    if args.task_type == "multilabel":
        # Multilabel case
        tgt_tensor = torch.stack([row[3] for row in batch])
    else:
        # Single Label case
        tgt_tensor = torch.cat([row[3] for row in batch]).long()

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
        return text_tensor, segment_tensor, mask_tensor, mm_mask_tensor, img_tensor, tgt_tensor
    else:
        return text_tensor, segment_tensor, mask_tensor, img_tensor, tgt_tensor


def get_data_loaders(args):
    '''
    tokenizer = (
        BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        if args.model in ["bert", "mmbt", "concatbert", "mmtr", "mmbtp"]
        else str.split
    )
    '''
    if args.model in ["bert", "mmbt", "concatbert", "mmtr", "mmbtp", "vilbert", "mmbt3", "mmvilbt"]:
        tokenizer = (
            BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )
    
    elif args.model in ["mmdbt", "mmbtrating"]:
        tokenizer = (
            DistilBertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
        )
    
    else:
        tokenizer = (str.split)

    transforms = get_transforms(args)

    args.labels, args.label_freqs = get_labels_and_frequencies(
        os.path.join(args.data_path, args.task, "train.jsonl")
    )
    vocab = get_vocab(args)
    args.vocab = vocab
    args.vocab_sz = vocab.vocab_sz
    args.n_classes = len(args.labels)

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
