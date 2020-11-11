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
from transformers import BertModel as huggingBertModel
from transformers import BertTokenizer as huggingBertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from mmbt.models.transformer import TransformerEncoder
from collections import OrderedDict
import math
import numpy as np


class MultimodalBertEncoder(nn.Module):
    def __init__(self, args):
        super(MultimodalBertEncoder, self).__init__()
        self.args = args
        
        self.tokbert = huggingBertModel.from_pretrained("bert-base-uncased")
        self.att_pooling = nn.Parameter(torch.rand(args.hidden_sz))
        
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings
            
        if self.args.pooling == 'cls_att':
            pooling_dim = 2*args.hidden_sz
        else:
            pooling_dim = args.hidden_sz

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
        
        #input_txt = input_txt[:, :min(input_txt.size(1), 10000)]
        
        # Process text in chunks
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
                
        for i in range(min(num_steps, 30)):
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
            
            out = self.tokbert(token_chunk_embeddings, extended_attention_mask)[0]
            #out = self.text2tok_lstm(token_chunk_embeddings)[0]
                                    
            dot = (out*self.att_pooling).sum(-1)  # Matrix of dot products (B, L)
            weights = F.softmax(dot, dim=1).unsqueeze(2)  # Normalize dot products and expand last dim (B, L, 1)
            weighted_sum = (out*weights).sum(dim=1)  # Weighted sum of hidden vectors (B, hidden_sz)
            
            chunk_tokens.append(weighted_sum)
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
                    
        attention_mask = torch.ones(bsz, txt_embed.size(1)).long().cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        segment = (
            torch.LongTensor(txt_embed.size(0), txt_embed.size(2))
            .fill_(1)
            .cuda()
        ).unsqueeze(1)
        
        seq_length = txt_embed.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.txt_embeddings.position_embeddings(position_ids)
        
        txt_embed_out = segment+txt_embed+position_embeddings
        
        # Output all encoded layers only for vertical attention on CLS token
        encoded_layers = self.encoder(
                txt_embed_out, extended_attention_mask, output_all_encoded_layers=(self.args.pooling == 'vert_att')
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
    
    
class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        
        self.bert = huggingBertModel.from_pretrained(args.bert_model)

    def forward(self, input_txt, attention_mask, segment):
        bsz = input_txt.size(0)
        
        #import pdb; pdb.set_trace()
        
        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        
        token_chunk_embeddings = torch.cat(
            [cls_id, input_txt[:, :510], sep_id], dim=1
        )
        
        out = self.bert(token_chunk_embeddings)
        
        return out[1]


class BertEncoderHierarchical(nn.Module):
    def __init__(self, args):
        super(BertEncoderHierarchical, self).__init__()
        self.args = args
        
        self.bert = huggingBertModel.from_pretrained(args.bert_model)
        
        bert = BertModel.from_pretrained(args.bert_model)
        self.word_embeddings = bert.embeddings.word_embeddings
        self.position_embeddings = nn.Embedding(512, args.hidden_sz)
        self.token_type_embeddings = nn.Embedding(2, args.hidden_sz)
        self.LayerNorm = BertLayerNorm(args.hidden_sz, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        self.att_pooling = nn.Parameter(torch.rand(args.hidden_sz))
        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def forward(self, input_txt, attention_mask, segment, input_img):
        
        bsz = input_txt.size(0)
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
        
        for i in range(min(30, num_steps)):
            cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
            cls_id = cls_id.unsqueeze(0).expand(bsz, 1)

            sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
            sep_id = sep_id.unsqueeze(0).expand(bsz, 1)

            start_idx = i*self.args.chunk_size
            end_idx = (i+1)*self.args.chunk_size
            
            token_chunk_embeddings = torch.cat(
                [cls_id, input_txt[:, start_idx:end_idx], sep_id], dim=1
            )
        
            out = self.bert(token_chunk_embeddings)[1]
            chunk_tokens.append(out)
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
        
        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)
        
        chunk_embeddings = torch.cat([cls_token_embeds, txt_embed, sep_token_embeds], dim=1)
        
        seq_length = chunk_embeddings.size(1)
                
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        
        token_type_ids = torch.zeros(bsz, seq_length, dtype=torch.long).cuda()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = chunk_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        attention_mask = torch.ones(bsz, seq_length).long().cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        encoded_layers = self.encoder(
                embeddings, extended_attention_mask
            )
        
        return self.pooler(encoded_layers[-1])
    

class BertEncoderHierarchicalOverlap(nn.Module):
    def __init__(self, args):
        super(BertEncoderHierarchicalOverlap, self).__init__()
        self.args = args
        
        self.bert = huggingBertModel.from_pretrained(args.bert_model)
        
        bert = BertModel.from_pretrained(args.bert_model)
        self.word_embeddings = bert.embeddings.word_embeddings
        self.position_embeddings = nn.Embedding(512, args.hidden_sz)
        self.token_type_embeddings = nn.Embedding(2, args.hidden_sz)
        self.LayerNorm = BertLayerNorm(args.hidden_sz, eps=1e-12)
        self.dropout = nn.Dropout(0.1)
        
        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def forward(self, input_txt, attention_mask, segment, input_img):
        
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
        
        for i in range(min(20, num_steps)):
            start_idx = i*self.args.chunk_size
            end_idx = (i+1)*self.args.chunk_size
            
            if i > 0:
                start_idx = start_idx - 10
        
            out = self.bert(input_txt[:, start_idx:end_idx])[1]
            chunk_tokens.append(out)
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
        
        bsz = txt_embed.size(0)
        
        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_token_embeds = self.word_embeddings(cls_id)

        sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
        sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
        sep_token_embeds = self.word_embeddings(sep_id)
        
        token_chunk_embeddings = torch.cat([cls_token_embeds, txt_embed, sep_token_embeds], dim=1)
        
        seq_length = token_chunk_embeddings.size(1)
                
        position_ids = torch.arange(seq_length, dtype=torch.long).cuda()
        position_ids = position_ids.unsqueeze(0).expand(bsz, seq_length)
        position_embeddings = self.position_embeddings(position_ids)
        
        token_type_ids = torch.zeros(bsz, seq_length, dtype=torch.long).cuda()
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_chunk_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        attention_mask = torch.ones(bsz, seq_length).long().cuda()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        encoded_layers = self.encoder(
                embeddings, extended_attention_mask
            )
        
        return self.pooler(encoded_layers[-1])

    
class TransEncoderHierarchicalPrev(nn.Module):
    def __init__(self, args):
        super(TransEncoderHierarchicalPrev, self).__init__()
        self.args = args
        
        self.bert = huggingBertModel.from_pretrained(args.bert_model)
        self.trans_l_mem = TransformerEncoder(embed_dim=768,
                                  num_heads=self.args.num_heads,
                                  layers=self.args.layers,
                                  attn_dropout=self.args.attn_dropout,
                                  relu_dropout=self.args.relu_dropout,
                                  res_dropout=self.args.res_dropout,
                                  embed_dropout=self.args.embed_dropout,
                                  attn_mask=self.args.attn_mask)

    def forward(self, input_txt, attention_mask, segment):
        
        bsz = input_txt.size(0)
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
        
        cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
        cls_id = cls_id.unsqueeze(0).expand(bsz, 1)
        cls_id = self.bert.embeddings(cls_id)
        
        for i in range(min(30, num_steps)):
            
            sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
            sep_id = sep_id.unsqueeze(0).expand(bsz, 1)
            sep_id = self.bert.embeddings(sep_id)

            start_idx = i*self.args.chunk_size
            end_idx = (i+1)*self.args.chunk_size
            
            txt_chunk = self.bert.embeddings(input_txt[:, start_idx:end_idx])
                        
            token_chunk_embeddings = torch.cat(
                [cls_id, txt_chunk, sep_id], dim=1
            )
            
            input_shape = token_chunk_embeddings.size()[:-1]
            device = token_chunk_embeddings.device
            attention_mask = torch.ones(input_shape, device=device)
            extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)
            encoder_extended_attention_mask = None
            head_mask = self.bert.get_head_mask(None, self.bert.config.num_hidden_layers)
        
            encoder_outputs = self.bert.encoder(token_chunk_embeddings,
                                                  attention_mask=extended_attention_mask,
                                                  head_mask=head_mask)
            sequence_output = encoder_outputs[0]
            pooled_output = self.bert.pooler(sequence_output)
            chunk_tokens.append(pooled_output)
            cls_id = sequence_output[:, 0]
            cls_id = cls_id.type(sep_id.type())
            cls_id = cls_id.unsqueeze(1)
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
        
        txt_embed = txt_embed.permute(1, 0, 2)

        h_ls = self.trans_l_mem(txt_embed)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        
        return h_ls[-1]
    

class TransEncoderHierarchical(nn.Module):
    def __init__(self, args):
        super(TransEncoderHierarchical, self).__init__()
        self.args = args
        
        self.bert = huggingBertModel.from_pretrained(args.bert_model)
        self.trans_l_mem = TransformerEncoder(embed_dim=768,
                                  num_heads=self.args.num_heads,
                                  layers=self.args.layers,
                                  attn_dropout=self.args.attn_dropout,
                                  relu_dropout=self.args.relu_dropout,
                                  res_dropout=self.args.res_dropout,
                                  embed_dropout=self.args.embed_dropout,
                                  attn_mask=self.args.attn_mask)

    def forward(self, input_txt, attention_mask, segment):
        
        bsz = input_txt.size(0)
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
        
        for i in range(min(30, num_steps)):
            cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
            cls_id = cls_id.unsqueeze(0).expand(bsz, 1)

            sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
            sep_id = sep_id.unsqueeze(0).expand(bsz, 1)

            start_idx = i*self.args.chunk_size
            end_idx = (i+1)*self.args.chunk_size
            
            token_chunk_embeddings = torch.cat(
                [cls_id, input_txt[:, start_idx:end_idx], sep_id], dim=1
            )
        
            out = self.bert(token_chunk_embeddings)[1]
            chunk_tokens.append(out)
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
        
        txt_embed = txt_embed.permute(1, 0, 2)

        h_ls = self.trans_l_mem(txt_embed)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        
        return h_ls[-1]
    
    
class TransEncoderHierarchicalTrope(nn.Module):
    def __init__(self, args):
        super(TransEncoderHierarchicalTrope, self).__init__()
        self.args = args
        
        self.bert = huggingBertModel.from_pretrained(args.bert_model)
        self.trans_l_mem = TransformerEncoder(embed_dim=768,
                                  num_heads=self.args.num_heads,
                                  layers=self.args.layers,
                                  attn_dropout=self.args.attn_dropout,
                                  relu_dropout=self.args.relu_dropout,
                                  res_dropout=self.args.res_dropout,
                                  embed_dropout=self.args.embed_dropout,
                                  attn_mask=self.args.attn_mask)
        
        with open('centroids.npy', 'rb') as f:
            centroids_load = np.load(f)
            
        #self.tropes_centroids = torch.nn.Parameter(torch.from_numpy(centroids_load), requires_grad=True).t().cuda()
        self.tropes_centroids = torch.from_numpy(centroids_load).t().cuda()

    def forward(self, input_txt, attention_mask, segment):
        
        bsz = input_txt.size(0)
        chunk_tokens = []
        num_steps = input_txt.size(1)//self.args.chunk_size + (input_txt.size(1)//self.args.chunk_size != 0)
        
        for i in range(min(30, num_steps)):
            cls_id = torch.LongTensor([self.args.vocab.stoi["[CLS]"]]).cuda()
            cls_id = cls_id.unsqueeze(0).expand(bsz, 1)

            sep_id = torch.LongTensor([self.args.vocab.stoi["[SEP]"]]).cuda()
            sep_id = sep_id.unsqueeze(0).expand(bsz, 1)

            start_idx = i*self.args.chunk_size
            end_idx = (i+1)*self.args.chunk_size
            
            token_chunk_embeddings = torch.cat(
                [cls_id, input_txt[:, start_idx:end_idx], sep_id], dim=1
            )
        
            out = self.bert(token_chunk_embeddings)[1]
            chunk_tokens.append(out)
            
        txt_embed = torch.stack(chunk_tokens, dim=1)
        
        txt_embed = txt_embed.permute(1, 0, 2)

        h_ls = self.trans_l_mem(txt_embed)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
            
        dot_prod = torch.matmul(h_ls[-1], self.tropes_centroids.to(txt_embed.device))
        
        return torch.cat([h_ls[-1], dot_prod], dim = 1)


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


class MultimodalBertRatingTextClf(nn.Module):
    def __init__(self, args):
        super(MultimodalBertRatingTextClf, self).__init__()
        self.args = args
        #self.enc = MultimodalBertEncoder(args)
        self.enc = BertEncoder(args)
        #self.enc = BertEncoderHierarchical(args)
        #self.enc = BertEncoderHierarchicalOverlap(args)
        #self.enc = TransEncoderHierarchicalPrev(args)
        #self.enc = TransEncoderHierarchicalTrope(args)
        
        #self.clf = nn.Linear(args.hidden_sz+len(args.genres), args.n_classes)
        #self.clf = SimpleClassifier(args.hidden_sz+len(args.genres), args.hidden_sz+len(args.genres), args.n_classes, 0.)
        #self.clf = nn.Linear(args.hidden_sz, args.n_classes)
        self.clf = SimpleClassifier(args.hidden_sz, args.hidden_sz, args.n_classes, 0.0)

    def forward(self, txt, mask, segment, img, genres=None):
        x = self.enc(txt, mask, segment)
        #input_cls = torch.cat((x, genres), dim=1)
        return self.clf(x)
