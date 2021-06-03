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
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME
from collections import OrderedDict

from mmbt.models.image import ImageEncoder


class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.args = args
        
        conv_layers = []
        conv_layers.append(nn.Conv1d(96, 96, 128, stride=2))
        conv_layers.append(nn.Conv1d(96, 96, 128, stride=2))
        conv_layers.append(nn.AdaptiveAvgPool1d(200))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class ModalityBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ModalityBertEmbeddings, self).__init__()
        self.args = args
        self.visual_embeddings = nn.Linear(args.orig_d_v, args.hidden_sz)
        self.audio_embeddings = nn.Linear(args.orig_d_a, args.hidden_sz)
        self.meta_embeddings = nn.Linear(312, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids, mod, cls_token=False):
        bsz = input_imgs.size(0)
        seq_length = token_type_ids.shape[1]

        if cls_token:
            cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
            cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
            cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)
        
        if mod == "vis":
            mod_embeddings = self.visual_embeddings(input_imgs)
        elif mod == "aud":
            mod_embeddings = self.audio_embeddings(input_imgs)
        elif mod == "meta":
            mod_embeddings = self.meta_embeddings(input_imgs)
        
        if cls_token:
            token_embeddings = torch.cat(
                [cls_token_embeds, mod_embeddings.unsqueeze(1), sep_token_embeds], dim=1
            )
        else:
            token_embeddings = torch.cat(
                [mod_embeddings.unsqueeze(1), sep_token_embeds], dim=1
            )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        if args.trained_model_dir:
            print("Loading BERT from AdaptaBERT pre-training")
            bert = BertModel.from_pretrained(args.trained_model_dir)
        else:
            bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        if args.task == "moviescope":
            ternary_embeds = nn.Embedding(3, args.hidden_sz)
            ternary_embeds.weight.data[:2].copy_(
                bert.embeddings.token_type_embeddings.weight
            )
            ternary_embeds.weight.data[2].copy_(
                bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)
            )
            self.txt_embeddings.token_type_embeddings = ternary_embeds
            
        if self.args.pooling == 'cls_att':
            pooling_dim = 2*args.hidden_sz
        else:
            pooling_dim = args.hidden_sz

        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.pooler_custom = nn.Sequential(
          nn.Linear(pooling_dim, args.hidden_sz),
          nn.Tanh(),
        )
        self.att_query = nn.Parameter(torch.rand(args.hidden_sz))
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, input_img):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([img_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
        
        # Output all encoded layers only for vertical attention on CLS token
        encoded_layers = self.encoder(
                encoder_input, extended_attention_mask, output_all_encoded_layers=(self.args.pooling == 'vert_att')
            )
        
        if self.args.pooling == 'cls':
            output = self.pooler(encoded_layers[-1])
        
        elif self.args.pooling == 'att':
            hidden = encoded_layers[-1]  # Get all hidden vectors of last layer (B, L, hidden_sz)
            dot = (hidden*self.att_query).sum(-1)  # Matrix of dot products (B, L)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, L, 1)
            weighted_sum = (hidden*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            output = self.pooler_custom(weighted_sum)
            
        elif self.args.pooling == 'cls_att':
            hidden = encoded_layers[-1]  # Get all hidden vectors of last layer (B, L, hidden_sz)
            cls_token = hidden[:, 0]  # Extract vector of CLS token
            word_tokens = hidden[:, 1:]
            dot = (word_tokens*self.att_query).sum(-1)  # Matrix of dot products (B, L)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, L, 1)
            weighted_sum = (word_tokens*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            pooler_cat = torch.cat([cls_token, weighted_sum], dim=1)
            output = self.pooler_custom(pooler_cat)
        
        else:
            hidden = [cls_hidden[:, 0] for cls_hidden in encoded_layers]  # Get all hidden vectors corresponding to CLS token (B, Num_bert_layers, hidden_sz)
            hidden = torch.stack(hidden, dim=1)  # Convert to tensor (B, Num_bert_layers, hidden_sz)
            dot = (hidden*self.att_query).sum(-1)  # Matrix of dot products (B, Num_bert_layers)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, Num_bert_layers, 1)
            weighted_sum = (hidden*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            output = self.pooler_custom(weighted_sum)

        return output


class MultimodalBertEncoder5M(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder5M, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        multi_embeds = nn.Embedding(6, args.hidden_sz)
        multi_embeds.weight.data[:2].copy_(
            bert.embeddings.token_type_embeddings.weight
        )
        multi_embeds.weight.data[2].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        multi_embeds.weight.data[3].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        multi_embeds.weight.data[4].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        multi_embeds.weight.data[5].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        self.txt_embeddings.token_type_embeddings = multi_embeds
            
        pooling_dim = args.hidden_sz

        self.mod_embeddings = ModalityBertEmbeddings(args, self.txt_embeddings)
        self.audio_encoder = AudioEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.pooler_custom = nn.Sequential(
          nn.Linear(pooling_dim, args.hidden_sz),
          nn.Tanh(),
        )
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, poster, video, audio, meta):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, 9).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(
        #    dtype=next(self.parameters()).dtype
        #)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        poster_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        vid_type_id = torch.LongTensor(bsz, 2).fill_(3).cuda()
        audio_type_id = torch.LongTensor(bsz, 2).fill_(4).cuda()
        meta_type_id = torch.LongTensor(bsz, 2).fill_(5).cuda()

        poster_embed_out = self.mod_embeddings(poster, poster_type_id, "vis", cls_token=True)
        vid_embed_out = self.mod_embeddings(torch.mean(video, dim=1), vid_type_id, "vis")
        audio_embed_out = self.mod_embeddings(torch.mean(self.audio_encoder(audio), dim=2), audio_type_id, "aud")
        meta_embed_out = self.mod_embeddings(meta, meta_type_id, "meta")
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([poster_embed_out, vid_embed_out, audio_embed_out, meta_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID
        
        encoded_layers = self.encoder(
                encoder_input, extended_attention_mask, output_all_encoded_layers=False
            )
        
        output = self.pooler(encoded_layers[-1])

        return output


class MultimodalBertEncoder4M(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder4M, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        multi_embeds = nn.Embedding(5, args.hidden_sz)
        multi_embeds.weight.data[:2].copy_(
            bert.embeddings.token_type_embeddings.weight
        )
        multi_embeds.weight.data[2].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        multi_embeds.weight.data[3].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        multi_embeds.weight.data[4].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        self.txt_embeddings.token_type_embeddings = multi_embeds
            
        pooling_dim = args.hidden_sz

        self.mod_embeddings = ModalityBertEmbeddings(args, self.txt_embeddings)
        self.audio_encoder = AudioEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.pooler_custom = nn.Sequential(
          nn.Linear(pooling_dim, args.hidden_sz),
          nn.Tanh(),
        )
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, poster, video, meta):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, 7).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(
        #    dtype=next(self.parameters()).dtype
        #)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        poster_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        vid_type_id = torch.LongTensor(bsz, 2).fill_(3).cuda()
        #audio_type_id = torch.LongTensor(bsz, 2).fill_(4).cuda()
        meta_type_id = torch.LongTensor(bsz, 2).fill_(4).cuda()

        poster_embed_out = self.mod_embeddings(poster, poster_type_id, "vis", cls_token=True)
        vid_embed_out = self.mod_embeddings(torch.mean(video, dim=1), vid_type_id, "vis")
        #audio_embed_out = self.mod_embeddings(torch.mean(self.audio_encoder(audio), dim=2), audio_type_id, "aud")
        meta_embed_out = self.mod_embeddings(meta, meta_type_id, "meta")
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([poster_embed_out, vid_embed_out, meta_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        encoded_layers = self.encoder(
                encoder_input, extended_attention_mask, output_all_encoded_layers=False
            )
        
        output = self.pooler(encoded_layers[-1])

        return output


class MultimodalBertEncoder3M(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder3M, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        multi_embeds = nn.Embedding(4, args.hidden_sz)
        multi_embeds.weight.data[:2].copy_(
            bert.embeddings.token_type_embeddings.weight
        )
        multi_embeds.weight.data[2].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        multi_embeds.weight.data[3].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        self.txt_embeddings.token_type_embeddings = multi_embeds
            
        pooling_dim = args.hidden_sz

        self.mod_embeddings = ModalityBertEmbeddings(args, self.txt_embeddings)
        self.audio_encoder = AudioEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.pooler_custom = nn.Sequential(
          nn.Linear(pooling_dim, args.hidden_sz),
          nn.Tanh(),
        )
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, video, audio):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, 5).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(
        #    dtype=next(self.parameters()).dtype
        #)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #poster_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        vid_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        audio_type_id = torch.LongTensor(bsz, 2).fill_(3).cuda()
        #meta_type_id = torch.LongTensor(bsz, 2).fill_(4).cuda()

        #poster_embed_out = self.mod_embeddings(poster, poster_type_id, "vis", cls_token=True)
        vid_embed_out = self.mod_embeddings(torch.mean(video, dim=1), vid_type_id, "vis", cls_token=True)
        audio_embed_out = self.mod_embeddings(torch.mean(self.audio_encoder(audio), dim=2), audio_type_id, "aud")
        #meta_embed_out = self.mod_embeddings(meta, meta_type_id, "meta")
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([vid_embed_out, audio_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        encoded_layers = self.encoder(
                encoder_input, extended_attention_mask, output_all_encoded_layers=False
            )
        
        output = self.pooler(encoded_layers[-1])

        return output


class MultimodalBertEncoder2M(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder2M, self).__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        multi_embeds = nn.Embedding(3, args.hidden_sz)
        multi_embeds.weight.data[:2].copy_(
            bert.embeddings.token_type_embeddings.weight
        )
        multi_embeds.weight.data[2].copy_(
            bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)+\
            torch.normal(mean=0, std=0.01, size=bert.embeddings.token_type_embeddings.weight.data.mean(dim=0).shape) 
        )
        self.txt_embeddings.token_type_embeddings = multi_embeds
            
        pooling_dim = args.hidden_sz

        self.mod_embeddings = ModalityBertEmbeddings(args, self.txt_embeddings)
        self.audio_encoder = AudioEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.pooler_custom = nn.Sequential(
          nn.Linear(pooling_dim, args.hidden_sz),
          nn.Tanh(),
        )
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, segment, audio):
        bsz = input_txt.size(0)
        attention_mask = torch.cat(
            [
                torch.ones(bsz, 3).long().cuda(),
                attention_mask,
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #extended_attention_mask = extended_attention_mask.to(
        #    dtype=next(self.parameters()).dtype
        #)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #poster_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        #vid_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        audio_type_id = torch.LongTensor(bsz, 3).fill_(2).cuda()
        #meta_type_id = torch.LongTensor(bsz, 2).fill_(4).cuda()

        #poster_embed_out = self.mod_embeddings(poster, poster_type_id, "vis", cls_token=True)
        #vid_embed_out = self.mod_embeddings(torch.mean(video, dim=1), vid_type_id, "vis", cls_token=True)
        audio_embed_out = self.mod_embeddings(torch.mean(self.audio_encoder(audio), dim=2), audio_type_id, "aud", cls_token=True)
        #meta_embed_out = self.mod_embeddings(meta, meta_type_id, "meta")
        txt_embed_out = self.txt_embeddings(input_txt, segment)
        encoder_input = torch.cat([audio_embed_out, txt_embed_out], 1)  # Bx(TEXT+IMG)xHID

        encoded_layers = self.encoder(
                encoder_input, extended_attention_mask, output_all_encoded_layers=False
            )
        
        output = self.pooler(encoded_layers[-1])

        return output


class MultimodalBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder2M(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, segment, vid, audio, poster, metadata):
        x = self.enc(txt, mask, segment, audio)
        return self.clf(x)
