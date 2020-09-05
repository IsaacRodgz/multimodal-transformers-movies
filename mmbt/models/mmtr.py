import torch
from torch import nn
import torch.nn.functional as F
from mmbt.models.image import ImageEncoder
from pytorch_pretrained_bert.modeling import BertModel
from mmbt.models.transformer import TransformerEncoder


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


class MMTransformerClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model.
        """
        super(MMTransformerClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 30, 30, 30
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
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
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
        return output