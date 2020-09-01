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
from mmbt.models.transformer import TransformerEncoder
from mmbt.models.image import ImageEncoder


class TextEncoder(nn.Module):
    def __init__(self, args, embeddings):
        super(TextEncoder, self).__init__()
        self.args = args
        self.word_embeddings = embeddings.word_embeddings
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, txt):
        bsz = txt.size(0)
        seq_length = txt.size(1) + 1  # +1 for SEP Token
        
        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)
        
        word_embeddings = self.word_embeddings(txt)
        token_embeddings = torch.cat(
            [sep_token_embeds, word_embeddings], dim=1
        )
        
        token_type_ids = (
            torch.LongTensor(bsz, seq_length)
            .fill_(1)
            .cuda()
        )
        
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super(ImageBertEmbeddings, self).__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, input_imgs, token_type_ids):
        bsz = input_imgs.size(0)
        seq_length = self.args.num_image_embeds + 2  # +2 for CLS and SEP Token

        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_embeddings = torch.cat(
            [cls_token_embeds, imgs_embeddings, sep_token_embeds], dim=1
        )

        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MMTransformer(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model.
        """
        super(MMTransformer, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_v = 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings
        self.img_encoder = ImageEncoder(args)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['v', 'lv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")
        
        return TransformerEncoder(embed_dim=embed_dim,
                                  num_heads=self.num_heads,
                                  layers=max(self.layers, layers),
                                  attn_dropout=attn_dropout,
                                  relu_dropout=self.relu_dropout,
                                  res_dropout=self.res_dropout,
                                  embed_dropout=self.embed_dropout,
                                  attn_mask=self.attn_mask)
    
    def forward(self, txt, img):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.txt_embeddings(txt)
 
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)

        # Project the textual/visual features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        # V --> L
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_l_with_vs = h_l_with_vs.transpose(0, 1)

        # L --> V
        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_ls = h_v_with_ls.transpose(0, 1)

        return h_l_with_vs, h_v_with_ls


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

        if args.model == "mmbt3":
            ternary_embeds = nn.Embedding(3, args.hidden_sz)
            ternary_embeds.weight.data[:2].copy_(
                bert.embeddings.token_type_embeddings.weight
            )
            ternary_embeds.weight.data[2].copy_(
                bert.embeddings.token_type_embeddings.weight.data.mean(dim=0)
            )
            self.txt_embeddings.token_type_embeddings = ternary_embeds

        self.transf = MMTransformer(args)
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.txt_encoder = TextEncoder(args, self.txt_embeddings)
        self.encoder = bert.encoder
        self.pooler = bert.pooler
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, input_txt, attention_mask, mm_attention_mask, input_img):
        
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        
        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        
        mm_txt, mm_img = self.transf(input_txt, img) # Get 3th multimodal input (mm_txt, mm_img)
        mm_embeddings_out = torch.cat([mm_img, mm_txt], dim=1)
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = self.txt_encoder(input_txt)
        '''
        print("Reached MultimodalBertEncoder")
        print("input_txt: ", input_txt.size())
        print("txt_embed_out: ", txt_embed_out.size())
        print("img_embed_out: ", img_embed_out.size())
        print("mm_embeddings_out: ", mm_embeddings_out.size())
        print("attention_mask: ", attention_mask.shape)
        print("mm_attention_mask: ", mm_attention_mask.shape)
        '''
        bsz = input_txt.size(0)
        txt_seq_len = input_txt.size(1) + 1
        img_seq_len = self.args.num_image_embeds + 2
        mm_max_seq_len = self.args.max_seq_len - img_seq_len - txt_seq_len
        mm_seq_len = min(mm_max_seq_len, mm_attention_mask.size(1))
        attention_mask = torch.cat(
            [
                torch.ones(bsz, img_seq_len).long().cuda(),
                attention_mask,
                mm_attention_mask[:,:mm_seq_len]
            ],
            dim=1,
        )
        
        #print("attention_mask: ", attention_mask.shape)
        
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        encoder_input = torch.cat([img_embed_out, txt_embed_out, mm_embeddings_out[:,:mm_seq_len]], 1)  # Bx(TEXT+IMG)xHID
        
        #print("mm_embeddings_out: ", encoder_input.size())
        #print("\n")

        encoded_layers = self.encoder(
            encoder_input, extended_attention_mask, output_all_encoded_layers=False
        )

        return self.pooler(encoded_layers[-1])


class MultimodalBertThreeClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertThreeClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        self.clf = nn.Linear(args.hidden_sz, args.n_classes)

    def forward(self, txt, mask, mm_mask, img):
        x = self.enc(txt, mask, mm_mask, img)
        return self.clf(x)
