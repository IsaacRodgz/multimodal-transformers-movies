import torch
from torch import nn
import torch.nn.functional as F
from mmbt.models.image import ImageEncoder
from pytorch_pretrained_bert.modeling import BertModel
from mmbt.models.transformer import TransformerEncoder
from transformers import DistilBertModel
from transformers import DistilBertTokenizer


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        encoded_layers, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
            output_all_encoded_layers=False,
        )
        return encoded_layers

    
class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.sigmoid_fc = nn.Linear(size_in1+size_in2, size_out, bias=False)

        # Activation functions
        self.tanh_f = nn.Tanh()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden1(x2))
        x_cat = torch.cat((x1, x2), dim=1)
        z = self.sigmoid_f(self.sigmoid_fc(x_cat))

        return z*h1 + (1-z)*h2, z


class MMTransformerRatingClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model as in the original paper.
        """
        super(MMTransformerRatingClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.img_encoder = ImageEncoder(args)

        combined_dim = self.d_l + self.d_v

        self.partial_mode = self.lonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l   # assuming d_l == d_v
        else:
            combined_dim = (self.d_l + self.d_v)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)
        
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.att_pooling = nn.Parameter(torch.rand(args.hidden_sz))

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        
        concat_dim = args.hidden_sz+len(args.genres)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, args.hidden_sz)
        
        self.clf = nn.Sequential(
            nn.Linear(concat_dim, concat_dim),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.BatchNorm1d(concat_dim, eps=1e-12),
            nn.Linear(concat_dim, output_dim),
        )

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
            
    def forward(self, input_txt, attention_mask, segment, img, genres):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        # Process text in chunks
        bsz = input_txt.size(0)
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
            
        x_l = torch.stack(chunk_tokens, dim=1)
        #x_l = self.enc(txt_embed, mask, segment)
        x_v = self.img_encoder(img)

        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.lonly:
            # V --> L
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = self.trans_l_mem(h_l_with_vs)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.vonly:
            # L --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_vs = self.trans_v_mem(h_v_with_ls)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 2:
            last_hs = torch.cat([last_h_l, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        
        input_cls = torch.cat((output, genres), dim=1)
        
        return self.clf(input_cls)


class MMTransformerUniBi(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that makes fusion of inputs with and
        without cross-attention: L + V + (L -> V) + (V -> L),
        i.e. fusion of four modalities.
        """
        super(MMTransformerUniBi, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.img_encoder = ImageEncoder(args)

        combined_dim = self.d_l + self.d_v

        self.partial_mode = self.lonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l   # assuming d_l == d_v
        else:
            combined_dim = (self.d_l + self.d_v)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            
        # 2. Self Attentions for uni-modal output (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_only = self.get_network(self_type='l_mem', layers=6)
        self.trans_v_only = self.get_network(self_type='v_mem', layers=6)
        
        # 3. Self Attentions for cross-modal output (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

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
            
    def forward(self, txt, mask, segment, img):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_v = self.img_encoder(img)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        
        # Unimodal processsing with transformer
        h_l_only = self.trans_l_only(proj_x_l)
        h_v_only = self.trans_v_only(proj_x_v)

        if self.lonly:
            # V --> L
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = self.trans_l_mem(h_l_with_vs)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
                
            # Combine text information with its cross-modal counterpart
            h_ls += h_l_only
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.vonly:
            # L --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_vs = self.trans_v_mem(h_v_with_ls)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
                
            # Combine image information with its cross-modal counterpart
            h_vs += h_v_only
            last_h_v = last_hs = h_vs[-1]
        
        if self.partial_mode == 2:
            last_hs = torch.cat([last_h_l, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output


class MMTransformerUniClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that processes only (V -> L) or (L -> V).
        """
        super(MMTransformerUniClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        self.img_encoder = ImageEncoder(args)
        
        combined_dim = self.d_l
        output_dim = args.n_classes

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        #if self.lonly:
        #    self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        #self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

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
            
    def forward(self, txt, mask, segment, img):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_v = self.img_encoder(img)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = x_v.transpose(1, 2)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        '''
        if self.lonly:
            # V --> L
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = self.trans_l_mem(h_l_with_vs)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
        '''
        if self.vonly:
            # L --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_vs = self.trans_v_mem(h_v_with_ls)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs = last_h_v
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output


class TransformerClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that processes only (V -> L) or (L -> V).
        """
        super(TransformerClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
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
        
        self.enc = BertEncoder(args)
        
        combined_dim = self.d_l
        output_dim = args.n_classes

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=8)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

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
            
    def forward(self, txt, mask, segment, img):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        h_ls = self.trans_l_mem(proj_x_l)
        if type(h_ls) == tuple:
            h_ls = h_ls[0]
        last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output


class MMTransformerGMUClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that, in the last layer, combines modalities
        through a GMU layer instead of concatenation.
        """
        super(MMTransformerGMUClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.combined_dim = 30
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
        
        self.enc = BertEncoder(args)
        self.img_encoder = ImageEncoder(args)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.args.n_classes)
        
        # GMU layer for fusing text and image information
        self.gmu = GatedMultimodalLayer(self.d_l, self.d_v, 30)

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
            
    def forward(self, txt, mask, segment, img, output_gate=False):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_v = self.img_encoder(img)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # Dimension (B, orig_d_l, L_txt)
        x_v = x_v.transpose(1, 2)  # Dimension (B, orig_v_l, L_img)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)  # Dimension (L_txt, B, d_l)
        proj_x_l = proj_x_l.permute(2, 0, 1)  # Dimension (L_img, B, d_v)

        if self.lonly:
            # V --> L
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)  # Dimension (L_txt, B, d_l)
            h_ls = self.trans_l_mem(h_l_with_vs)  # Dimension (L_txt, B, d_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]  # Dimension (B, d_l)

        if self.vonly:
            # L --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)  # Dimension (L_img, B, d_v)
            h_vs = self.trans_v_mem(h_v_with_ls)  # Dimension (L_img, B, d_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]  # Dimension (B, d_v)

        last_hs, z = self.gmu(last_h_l, last_h_v)
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)