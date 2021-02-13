import torch
from torch import nn
import torch.nn.functional as F
from mmbt.models.image import ImageEncoder
from pytorch_pretrained_bert.modeling import BertModel
from transformers import BertModel as huggingBertModel
from mmbt.models.transformer import TransformerEncoder


class AudioEncoderLarge(nn.Module):
    def __init__(self, args):
        super(AudioEncoderLarge, self).__init__()
        self.args = args
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, 128, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        '''
        conv_layers = []
        conv_layers.append(nn.Conv1d(128, 128, 128, stride=2))
        conv_layers.append(nn.Conv1d(128, 128, 128, stride=2))
        conv_layers.append(nn.AdaptiveAvgPool1d(200))
        self.conv_layers = nn.ModuleList(conv_layers)
        '''

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        #x = self.conv4(x)
        return x
    
    
class AudioEncoder(nn.Module):
    def __init__(self, args):
        super(AudioEncoder, self).__init__()
        self.args = args
        
        conv_layers = []
        '''
        conv_layers.append(
                nn.Sequential(
                nn.Conv1d(96, 96, 128, stride=2),
                nn.BatchNorm1d(96),
                nn.ReLU(),
                #nn.MaxPool1d(kernel_size=2)
            )
        )
        conv_layers.append(
                nn.Sequential(
                nn.Conv1d(96, 96, 128, stride=2),
                nn.BatchNorm1d(96),
                nn.ReLU(),
                #nn.MaxPool1d(kernel_size=2)
            )
        )
        conv_layers.append(nn.AdaptiveAvgPool1d(200))
        '''
        conv_layers.append(nn.Conv1d(96, 96, 128, stride=2))
        conv_layers.append(nn.Conv1d(96, 96, 128, stride=2))
        conv_layers.append(nn.AdaptiveAvgPool1d(200))
        self.conv_layers = nn.ModuleList(conv_layers)

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x


class BertEncoder(nn.Module):
    def __init__(self, args):
        super(BertEncoder, self).__init__()
        self.args = args
        self.bert = huggingBertModel.from_pretrained(args.bert_model)

    def forward(self, txt, mask, segment):
        encoded_layers, out = self.bert(
            txt,
            token_type_ids=segment,
            attention_mask=mask,
        )
        return encoded_layers

    
class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.x_gate = nn.Linear(size_in1+size_in2, size_out, bias=False)

    def forward(self, x1, x2):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        x_cat = torch.cat((x1, x2), dim=1)
        z = F.sigmoid(self.x_gate(x_cat))

        return z*h1 + (1-z)*h2, z


class TextShifting3Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_out):
        super(TextShifting3Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_out = size_in1, size_in2, size_in3, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3, size_out, bias=False)

    def forward(self, x1, x2, x3):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        x_cat = torch.cat((x1, x2, x3), dim=1)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3, torch.cat((z1, z2, z3), dim=1)


class TextShifting4Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_out):
        super(TextShifting4Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_out = size_in1, size_in2, size_in3, size_in4, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden4 = nn.Linear(size_in4, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)
        self.x4_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4, size_out, bias=False)

    def forward(self, x1, x2, x3, x4):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        h4 = F.tanh(self.hidden4(x4))
        x_cat = torch.cat((x1, x2, x3, x4), dim=1)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))
        z4 = F.sigmoid(self.x4_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3 + z4*h4, torch.cat((z1, z2, z3, z4), dim=1)


class TextShifting5Layer(nn.Module):
    """ Layer inspired by 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_in3, size_in4, size_in5, size_out):
        super(TextShifting5Layer, self).__init__()
        self.size_in1, self.size_in2, self.size_in3, self.size_in4, self.size_in5, self.size_out = size_in1, size_in2, size_in3, size_in4, size_in5, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden3 = nn.Linear(size_in3, size_out, bias=False)
        self.hidden4 = nn.Linear(size_in4, size_out, bias=False)
        self.hidden5 = nn.Linear(size_in5, size_out, bias=False)
        self.x1_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x2_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x3_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x4_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)
        self.x5_gate = nn.Linear(size_in1+size_in2+size_in3+size_in4+size_in5, size_out, bias=False)

    def forward(self, x1, x2, x3, x4, x5):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        h3 = F.tanh(self.hidden3(x3))
        h4 = F.tanh(self.hidden4(x4))
        h5 = F.tanh(self.hidden5(x5))
        x_cat = torch.cat((x1, x2, x3, x4, x5), dim=1)
        z1 = F.sigmoid(self.x1_gate(x_cat))
        z2 = F.sigmoid(self.x2_gate(x_cat))
        z3 = F.sigmoid(self.x3_gate(x_cat))
        z4 = F.sigmoid(self.x4_gate(x_cat))
        z5 = F.sigmoid(self.x5_gate(x_cat))

        return z1*h1 + z2*h2 + z3*h3 + z4*h4 + z5*h5, torch.cat((z1, z2, z3, z4, z5), dim=1)


class MMTransformerGMUVPAClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with GMU late fusion.
        """
        super(MMTransformerGMUVPAClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a = args.orig_d_l, args.orig_d_v, args.orig_d_a
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        combined_dim = self.d_l + self.d_a + self.d_v
        
        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2*self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2*(self.d_l + self.d_a + self.d_v)
        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # GMU layer for fusing text and image and audio information
        self.gmu = TextShifting3Layer(self.d_l*2, self.d_v*2, self.d_a*2, self.d_l)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #print(audio.shape)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output


class MMTransformerConcatVPAClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames and Audio spectrogram with Concat late fusion.
        """
        super(MMTransformerConcatVPAClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a = args.orig_d_l, args.orig_d_v, args.orig_d_a
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask

        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)

        combined_dim = self.d_l + self.d_a + self.d_v

        self.partial_mode = self.lonly + self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = 2*self.d_l   # assuming d_l == d_a == d_v
        else:
            combined_dim = 2*(self.d_l + self.d_a + self.d_v)

        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)

        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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

    def forward(self, txt, mask, segment, img, audio):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]

        if self.partial_mode == 3:
            last_hs = torch.cat([last_h_l, last_h_a, last_h_v], dim=1)

        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        output = self.out_layer(last_hs_proj)
        return output


class MMTransformerClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model as in the original paper.
        """
        super(MMTransformerClf, self).__init__()
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
    
    
class MMTransformerMoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text and Video frames with Concatenation late fusion.
        """
        super(MMTransformerMoviescopeClf, self).__init__()
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
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)

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


class MMTransformerGMUMoviescopeVidTextClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text and Video frames with GMU late fusion.
        """
        super(MMTransformerGMUMoviescopeVidTextClf, self).__init__()
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

        combined_dim = self.d_l + self.d_v

        self.partial_mode = self.lonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_l   # assuming d_l == d_v
        else:
            combined_dim = (self.d_l + self.d_v)
        combined_dim = 768 # For GMU
        
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
        
        # GMU layer for fusing text and video information
        self.gmu = GatedMultimodalLayer(self.d_l, self.d_v, self.d_l)

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
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)

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
        
        last_hs, z = self.gmu(last_h_l, last_h_v)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerGMUMoviescopeVidAudClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Audio and Video frames with GMU late fusion.
        """
        super(MMTransformerGMUMoviescopeVidAudClf, self).__init__()
        self.args = args
        self.orig_d_a, self.orig_d_v = args.orig_d_a, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.vonly = args.vonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.audio_enc = AudioEncoder(args)
        
        combined_dim = self.d_a + self.d_v

        self.partial_mode = self.aonly + self.vonly
        if self.partial_mode == 1:
            combined_dim = self.d_a   # assuming d_l == d_v
        else:
            combined_dim = (self.d_a + self.d_v)
        combined_dim = 768 # For GMU
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # GMU layer for fusing text and video information
        self.gmu = GatedMultimodalLayer(self.d_a, self.d_v, self.d_v)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
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
            
    def forward(self, img, audio, output_gate=False):
        """
        audio and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_a = self.audio_enc(audio)
        x_v = img.transpose(1, 2)
        
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.aonly:
            # V --> A
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = self.trans_a_mem(h_a_with_vs)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # A --> V
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = self.trans_v_mem(h_v_with_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs, z = self.gmu(last_h_a, last_h_v)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcatMoviescopeVidAudClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Audio and Video frames with Concat late fusion.
        """
        super(MMTransformerConcatMoviescopeVidAudClf, self).__init__()
        self.args = args
        self.orig_d_a, self.orig_d_v = args.orig_d_a, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.vonly = args.vonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.audio_enc = AudioEncoder(args)
        
        combined_dim = self.d_a + self.d_v

        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
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
            
    def forward(self, img, audio, output_gate=False):
        """
        audio and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_a = self.audio_enc(audio)
        x_v = img.transpose(1, 2)
        
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.aonly:
            # V --> A
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = self.trans_a_mem(h_a_with_vs)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # A --> V
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = self.trans_v_mem(h_v_with_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs = torch.cat([last_h_a, last_h_v], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        return self.out_layer(last_hs_proj)


class MMTransformerGMUMoviescopeTxtAudClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text and Audio frames with GMU late fusion.
        """
        super(MMTransformerGMUMoviescopeTxtAudClf, self).__init__()
        self.args = args
        self.orig_d_a, self.orig_d_l = args.orig_d_a, args.orig_d_l
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        combined_dim = 768

        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # GMU layer for fusing text and video information
        self.gmu = GatedMultimodalLayer(self.d_l, self.d_a, self.d_l)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'la']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['l', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, audio, output_gate=False):
        """
        audio and text should have dimension [batch_size, seq_len, n_features]
        """
        x_a = self.audio_enc(audio)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.aonly:
            # L --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_as = self.trans_a_mem(h_a_with_ls)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1]

        if self.lonly:
            # A --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_ls = self.trans_l_mem(h_l_with_as)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1]
        
        last_hs, z = self.gmu(last_h_l, last_h_a)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcatMoviescopeTxtAudClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text and Audio frames with Concat late fusion.
        """
        super(MMTransformerConcatMoviescopeTxtAudClf, self).__init__()
        self.args = args
        self.orig_d_a, self.orig_d_l = args.orig_d_a, args.orig_d_l
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        combined_dim = self.d_a + self.d_l

        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
       
        # Projection layers
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'la']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['l', 'al']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, audio):
        """
        audio and text should have dimension [batch_size, seq_len, n_features]
        """
        x_a = self.audio_enc(audio)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        if self.aonly:
            # L --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_as = self.trans_a_mem(h_a_with_ls)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = h_as[-1]

        if self.lonly:
            # A --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
            h_ls = self.trans_l_mem(h_l_with_as)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = h_ls[-1]
        
        last_hs = torch.cat([last_h_a, last_h_l], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        return self.out_layer(last_hs_proj)


class MMTransformerGMUMoviescopeVidAudPosterClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Poster image, Audio and Video frames with GMU late fusion.
        """
        super(MMTransformerGMUMoviescopeVidAudPosterClf, self).__init__()
        self.args = args
        self.orig_d_a, self.orig_d_v = args.orig_d_a, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.vonly = args.vonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        combined_dim = 768 # For GMU
        output_dim = args.n_classes
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)
        
        # GMU layer for fusing text and video information
        self.gmu = TextShifting3Layer(self.d_a, self.d_v, self.d_v, self.d_v)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
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
            
    def forward(self, img, audio, poster, output_gate=False):
        """
        audio and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_a = self.audio_enc(audio)
        x_v = img.transpose(1, 2)
        
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.aonly:
            # V --> A
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = self.trans_a_mem(h_a_with_vs)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # A --> V
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = self.trans_v_mem(h_v_with_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs, z = self.gmu(last_h_a, last_h_v, self.proj_poster(poster))
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcatMoviescopeVidAudPosterClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Poster image, Audio and Video frames with Concat late fusion.
        """
        super(MMTransformerConcatMoviescopeVidAudPosterClf, self).__init__()
        self.args = args
        self.orig_d_a, self.orig_d_v = args.orig_d_a, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.vonly = args.vonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)

        # 1. Temporal convolutional layers
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.aonly:
            self.trans_a_with_v = self.get_network(self_type='av')
        if self.vonly:
            self.trans_v_with_a = self.get_network(self_type='va')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
        # Projection layers
        combined_dim = self.d_v+self.d_a+self.d_v
        output_dim = args.n_classes
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['a', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'a_mem':
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
            
    def forward(self, img, audio, poster):
        """
        audio and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_a = self.audio_enc(audio)
        x_v = img.transpose(1, 2)
        
        # Project the visual/audio features
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.aonly:
            # V --> A
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = self.trans_a_mem(h_a_with_vs)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # A --> V
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = self.trans_v_mem(h_v_with_as)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
            
        last_hs = torch.cat([last_h_a, last_h_v, self.proj_poster(poster)], dim=1)
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        return self.out_layer(last_hs_proj)


class MMTransformerGMUMoviescopeVidAudPosterTxtClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, Audio spectrogram and poster with GMU late fusion.
        """
        super(MMTransformerGMUMoviescopeVidAudPosterTxtClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, self.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.combined_dim = 768
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)
        
        # GMU layer for fusing text and image information
        self.gmu = TextShifting4Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_l)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster))
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcatMoviescopeVidAudPosterTxtClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, Audio spectrogram and poster with Concat late fusion.
        """
        super(MMTransformerConcatMoviescopeVidAudPosterTxtClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, self.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.combined_dim = self.d_l*2+self.d_a*2+self.d_v*2+self.d_v
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs = torch.cat([last_h_l,
                             last_h_a,
                             last_h_v,
                             self.proj_poster(poster)], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        return self.out_layer(last_hs_proj)


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
        if self.lonly:
            self.trans_l_with_v = self.get_network(self_type='lv')
        #if self.vonly:
        #    self.trans_v_with_l = self.get_network(self_type='vl')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        #self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
       
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
        '''
        if self.vonly:
            # L --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_vs = self.trans_v_mem(h_v_with_ls)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        '''
        last_hs = last_h_l
        
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
    
    
class TransformerVideoClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that processes only (V -> L) or (L -> V).
        """
        super(TransformerVideoClf, self).__init__()
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
                
        combined_dim = self.d_l
        output_dim = args.n_classes

        # 1. Temporal convolutional layers
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=8)
       
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
        x_v = F.dropout(img.transpose(1, 2), p=self.embed_dropout, training=self.training)

        # Project the textual/visual/audio features
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        h_ls = self.trans_v_mem(proj_x_v)
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
        
        
class MMTransformerGMUMoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that, in the last layer, combines modalities
        through a GMU layer instead of concatenation. Support for text-video-poster info
        """
        super(MMTransformerGMUMoviescopeClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v = args.orig_d_l, args.orig_d_v
        self.d_l, self.d_a, self.d_v = 768, 768, 768
        self.combined_dim = 768
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
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)

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
        self.gmu = TextShifting3Layer(self.d_l, self.d_v, self.d_v, self.d_l)

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
            
    def forward(self, txt, mask, segment, img, poster, output_gate=False):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # Dimension (B, orig_d_l, L_txt)
        x_v = img.transpose(1, 2)  # Dimension (B, orig_v_l, L_img)

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

        last_hs, z = self.gmu(last_h_l, last_h_v, self.proj_poster(poster))
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcatMoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model that, in the last layer, combines modalities
        through concatenation. Support for text-video-poster info.
        """
        super(MMTransformerConcatMoviescopeClf, self).__init__()
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
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)

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
        self.combined_dim = self.d_l+self.d_v+self.d_v
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.args.n_classes)
        
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
            
    def forward(self, txt, mask, segment, img, poster, output_gate=False):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # Dimension (B, orig_d_l, L_txt)
        x_v = img.transpose(1, 2)  # Dimension (B, orig_v_l, L_img)

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

        last_hs = torch.cat([last_h_l, last_h_v, self.proj_poster(poster)], dim=1)
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs

        return self.out_layer(last_hs_proj)


class MMTransformerGMU4MoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, poster and metadata with GMU late fusion.
        """
        super(MMTransformerGMU4MoviescopeClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_m = args.orig_d_l, args.orig_d_v, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
        self.combined_dim = 768
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
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        # 0. Project metadata feature to 768 dim
        self.proj_metadata = nn.Linear(self.orig_d_m, self.d_m)

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
        self.gmu = TextShifting4Layer(self.d_l, self.d_v, self.d_v, self.d_m, self.d_l)

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
            
    def forward(self, txt, mask, segment, img, poster, metadata, output_gate=False):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # Dimension (B, orig_d_l, L_txt)
        x_v = img.transpose(1, 2)  # Dimension (B, orig_v_l, L_img)

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

        last_hs, z = self.gmu(last_h_l, last_h_v, self.proj_poster(poster), self.proj_metadata(metadata))
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcat4MoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, poster and metadata with Concatenation late fusion.
        """
        super(MMTransformerConcat4MoviescopeClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_m = args.orig_d_l, args.orig_d_v, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
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
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        # 0. Project metadata feature to 768 dim
        self.proj_metadata = nn.Linear(self.orig_d_m, self.d_m)

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
        self.combined_dim = self.d_l+self.d_v+self.d_v+self.d_m
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, self.args.n_classes)

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
            
    def forward(self, txt, mask, segment, img, poster, metadata, output_gate=False):
        """
        text, and vision should have dimension [batch_size, seq_len, n_features]
        """
        x_l = self.enc(txt, mask, segment)
        '''
        seg_features = []
        for i in range(self.args.num_images):
            seg_features.append(self.img_encoder(img[:,i,...]))
        x_v = torch.cat(seg_features, dim=1)
        '''
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)  # Dimension (B, orig_d_l, L_txt)
        x_v = img.transpose(1, 2)  # Dimension (B, orig_v_l, L_img)

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
            
        
            
        last_hs = torch.cat([last_h_l,
                             last_h_v,
                             self.proj_poster(poster),
                             self.proj_metadata(metadata)], dim=1)
                
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerConcat5MoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, Audio spectrogram, poster and metadata with Concatenation late fusion.
        """
        super(MMTransformerConcat5MoviescopeClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, self.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        # 0. Project metadata feature to 768 dim
        self.proj_metadata = nn.Linear(self.orig_d_m, self.d_m)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.combined_dim = self.d_l*2+self.d_v*2+self.d_a*2+self.d_v+self.d_m
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, metadata):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #print(audio.shape)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs = torch.cat([last_h_l,
                             last_h_v,
                             last_h_a,
                             self.proj_poster(poster),
                             self.proj_metadata(metadata)], dim=1)
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
        
        output = self.out_layer(last_hs_proj)
        return output


class MMTransformerGMU5MoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, Audio spectrogram, poster and metadata with Concatenation late fusion.
        """
        super(MMTransformerGMU5MoviescopeClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, self.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        # 0. Project metadata feature to 768 dim
        self.proj_metadata = nn.Linear(self.orig_d_m, self.d_m)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)

        # 2. Crossmodal Attentions
        if self.lonly:
            self.trans_l_with_a = self.get_network(self_type='la')
            self.trans_l_with_v = self.get_network(self_type='lv')
        if self.vonly:
            self.trans_v_with_l = self.get_network(self_type='vl')
            self.trans_v_with_a = self.get_network(self_type='va')
        if self.aonly:
            self.trans_a_with_l = self.get_network(self_type='al')
            self.trans_a_with_v = self.get_network(self_type='av')
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.combined_dim = 768
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)
        
        # GMU layer for fusing text and image information
        self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_m, self.d_l)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = 2*self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = 2*self.d_a, self.attn_dropout
        elif self_type == 'v_mem':
            embed_dim, attn_dropout = 2*self.d_v, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, metadata, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #print(audio.shape)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)    # Dimension (L, N, d_l)
            h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)    # Dimension (L, N, d_l)
            h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
            h_ls = self.trans_l_mem(h_ls)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
            h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
            h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
            h_as = self.trans_a_mem(h_as)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
            h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
            h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
            h_vs = self.trans_v_mem(h_vs)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster), self.proj_metadata(metadata))
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)


class MMTransformerGMU5IntraMoviescopeClf(nn.Module):
    def __init__(self, args):
        """
        Construct a MulT model for Text, Video frames, Audio spectrogram, poster and metadata with Concatenation late fusion.
        """
        super(MMTransformerGMU5MoviescopeClf, self).__init__()
        self.args = args
        self.orig_d_l, self.orig_d_v, self.orig_d_a, self.orig_d_m = args.orig_d_l, args.orig_d_v, args.orig_d_a, 312
        self.d_l, self.d_a, self.d_v, self.d_m = 768, 768, 768, 768
        self.vonly = args.vonly
        self.lonly = args.lonly
        self.aonly = args.aonly
        self.num_heads = args.num_heads
        self.layers = args.layers
        self.attn_dropout = args.attn_dropout
        self.attn_dropout_v = args.attn_dropout_v
        self.attn_dropout_a = args.attn_dropout_a
        self.relu_dropout = args.relu_dropout
        self.res_dropout = args.res_dropout
        self.out_dropout = args.out_dropout
        self.embed_dropout = args.embed_dropout
        self.attn_mask = args.attn_mask
        
        self.enc = BertEncoder(args)
        self.audio_enc = AudioEncoder(args)
        
        # 0. Project poster feature to 768 dim
        self.proj_poster = nn.Linear(self.orig_d_v, self.d_v)
        
        # 0. Project metadata feature to 768 dim
        self.proj_metadata = nn.Linear(self.orig_d_m, self.d_m)
        
        output_dim = args.n_classes        # This is actually not a hyperparameter :-)

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        
        # 3. Self Attentions (Could be replaced by LSTMs, GRUs, etc.)
        #    [e.g., self.trans_x_mem = nn.LSTM(self.d_x, self.d_x, 1)
        self.trans_l_mem = self.get_network(self_type='l_mem', layers=3)
        self.trans_v_mem = self.get_network(self_type='v_mem', layers=3)
        self.trans_a_mem = self.get_network(self_type='a_mem', layers=3)
       
        # Projection layers
        self.combined_dim = 768
        self.proj1 = nn.Linear(self.combined_dim, self.combined_dim)
        self.proj2 = nn.Linear(self.combined_dim, self.combined_dim)
        self.out_layer = nn.Linear(self.combined_dim, output_dim)
        
        # GMU layer for fusing text and image information
        self.gmu = TextShifting5Layer(self.d_l*2, self.d_v*2, self.d_v*2, self.d_v, self.d_m, self.d_l)
        
    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'al', 'vl']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'la', 'va']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'lv', 'av']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == 'l_mem':
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type == 'a_mem':
            embed_dim, attn_dropout = self.d_a, self.attn_dropout
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
            
    def forward(self, txt, mask, segment, img, audio, poster, metadata, output_gate=False):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        #print(audio.shape)
        x_l = self.enc(txt, mask, segment)
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_v = img.transpose(1, 2)
        x_a = self.audio_enc(audio)

        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_l = proj_x_l.permute(2, 0, 1)
        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)

        if self.lonly:
            # (V,A) --> L
            h_ls = self.trans_l_mem(proj_x_l)
            if type(h_ls) == tuple:
                h_ls = h_ls[0]
            last_h_l = last_hs = h_ls[-1]   # Take the last output for prediction

        if self.aonly:
            # (L,V) --> A
            h_as = self.trans_a_mem(proj_x_a)
            if type(h_as) == tuple:
                h_as = h_as[0]
            last_h_a = last_hs = h_as[-1]

        if self.vonly:
            # (L,A) --> V
            h_vs = self.trans_v_mem(proj_x_v)
            if type(h_vs) == tuple:
                h_vs = h_vs[0]
            last_h_v = last_hs = h_vs[-1]
        
        last_hs, z = self.gmu(last_h_l, last_h_v, last_h_a, self.proj_poster(poster), self.proj_metadata(metadata))
        
        # A residual block
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training))
        last_hs_proj += last_hs
                
        if output_gate:
            return self.out_layer(last_hs_proj), z
        else:
            return self.out_layer(last_hs_proj)