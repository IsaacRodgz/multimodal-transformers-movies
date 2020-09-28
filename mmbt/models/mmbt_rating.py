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
from transformers import DistilBertModel
from transformers import DistilBertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME
from collections import OrderedDict
import math

from mmbt.models.image import ImageEncoder


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


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.att_pooling = nn.Parameter(torch.rand(args.hidden_sz))
        
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings

        if args.task == "vsnli":
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
        
        # Process text in chunks
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
                
        for i in range(num_steps):
            cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
            cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
            
            sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
            sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
            
            start_idx = i*self.args.chunk_size
            end_idx = (i+1)*self.args.chunk_size
            
            
            token_chunk_embeddings = torch.cat(
                [cls_id, input_txt[:, start_idx:end_idx], sep_id], dim=1
            )
            
            extended_attention_mask = torch.cat(
                [
                    torch.ones(bsz, 1).long().cuda(),
                    attention_mask[:, start_idx:end_idx],
                    torch.ones(bsz, 1).long().cuda(),
                ],
                dim=1,
            )
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )
            
            out = self.distilbert(token_chunk_embeddings, extended_attention_mask)[0]
            #out = self.text2tok_lstm(token_chunk_embeddings)[0]
                                    
            dot = (out*self.att_pooling).sum(-1)  # Matrix of dot products (B, L)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, L, 1)
            weighted_sum = (out*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            
            chunk_tokens.append(weighted_sum.detach())
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
                    
        attention_mask = torch.cat(
            [
                torch.ones(bsz, self.args.num_image_embeds + 2).long().cuda(),
                torch.ones(bsz, txt_embed.size(1)).long().cuda(),
            ],
            dim=1,
        )
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        img_tok = (
            torch.LongTensor(txt_embed.size(0), self.args.num_image_embeds + 2)
            .fill_(0)
            .cuda()
        )
        segment = (
            torch.LongTensor(txt_embed.size(0), txt_embed.size(2))
            .fill_(1)
            .cuda()
        ).unsqueeze(1)
        
        img = self.img_encoder(input_img)  # BxNx3x224x224 -> BxNx2048
        img_embed_out = self.img_embeddings(img, img_tok)
        txt_embed_out = segment+txt_embed
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
    
    
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)
    
    
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
    
class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super().__init__()
        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.Dropout(dropout),
            GeLU(),
            BertLayerNorm(hid_dim, eps=1e-12),
            nn.Linear(hid_dim, out_dim),
        )

    def forward(self, hidden_states):
        return self.logit_fc(hidden_states)


class MultimodalBertRatingClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertRatingClf, self).__init__()
        self.args = args
        self.enc = MultimodalBertEncoder(args)
        #self.clf = nn.Linear(args.hidden_sz+len(args.genres), args.n_classes)
        self.clf = SimpleClassifier(args.hidden_sz+len(args.genres), args.hidden_sz+len(args.genres), args.n_classes, 0.)

    def forward(self, txt, mask, segment, img, genres):
        x = self.enc(txt, mask, segment, img)
        input_cls = torch.cat((x, genres), dim=1)
        return self.clf(input_cls)
