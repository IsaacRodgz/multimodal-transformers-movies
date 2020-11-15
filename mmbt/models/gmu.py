import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from mmbt.models.bow import GloveBowEncoder
from mmbt.models.image import ImageEncoder16


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class GatedMultimodalLayer(nn.Module):
    """ Gated Multimodal Layer based on 'Gated multimodal networks, Arevalo1 et al.' (https://arxiv.org/abs/1702.01992) """
    def __init__(self, size_in1, size_in2, size_out):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out
        
        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_in1+size_in2, size_out, bias=False)

    def forward(self, x1, x2):
        h1 = F.tanh(self.hidden1(x1))
        h2 = F.tanh(self.hidden2(x2))
        x = torch.cat((x1, x2), dim=1)
        z = F.sigmoid(self.hidden_sigmoid(x))

        return z*h1 + (1-z)*h2
    
    
class MaxOut(nn.Module):
    def __init__(self, input_dim, output_dim, num_units=2):
        super(MaxOut, self).__init__()
        self.fc1_list = nn.ModuleList([nn.Linear(input_dim, output_dim) for i in range(num_units)])

    def forward(self, x): 

        return self.maxout(x, self.fc1_list)

    def maxout(self, x, layer_list):
        max_output = layer_list[0](x)
        for _, layer in enumerate(layer_list, start=1):
            max_output = torch.max(max_output, layer(x))
        return max_output


class MLPGenreClassifierModel(nn.Module):

    def __init__(self, args, concat=False):

        super(MLPGenreClassifierModel, self).__init__()
        self.args = args
        
        if not concat:
            self.bn1 = nn.BatchNorm1d(args.hidden_sz)
            self.linear1 = MaxOut(args.hidden_sz, args.hidden_sz)
        else:
            self.bn1 = nn.BatchNorm1d(args.embed_sz+args.img_hidden_sz)
            self.linear1 = MaxOut(args.embed_sz+args.img_hidden_sz, args.hidden_sz)
        self.drop1 = nn.Dropout(p=args.dropout)
        
        self.bn2 = nn.BatchNorm1d(args.hidden_sz)
        self.linear2 = MaxOut(args.hidden_sz, args.hidden_sz)
        self.drop2 = nn.Dropout(p=args.dropout)
        
        self.bn3 = nn.BatchNorm1d(args.hidden_sz)
        self.linear3 = nn.Linear(args.hidden_sz, args.n_classes)
        #self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, feature_images=None):
        if feature_images is None:
            x = input_ids
        else:
            x = torch.cat([input_ids, feature_images], dim=1)
        x = self.bn1(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.bn2(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.bn3(x)
        x = self.linear3(x)

        return x

    
class GMUClf(nn.Module):

    def __init__(self, args):

        super(GMUClf, self).__init__()
        self.args = args
        
        self.txtenc = GloveBowEncoder(args)
        self.imgenc = ImageEncoder16(args)
        
        self.visual_mlp = torch.nn.Sequential(
            nn.BatchNorm1d(args.img_hidden_sz),
            nn.Linear(args.img_hidden_sz, args.hidden_sz)
        )
        self.textual_mlp = torch.nn.Sequential(
            nn.BatchNorm1d(args.embed_sz),
            nn.Linear(args.embed_sz, args.hidden_sz)
        )
        
        self.gmu = GatedMultimodalLayer(args.hidden_sz, args.hidden_sz, args.hidden_sz)
        
        self.logistic_mlp = MLPGenreClassifierModel(args)

    def forward(self, txt, img):
        
        txt = self.txtenc(txt)
        x_t = self.textual_mlp(txt)
        img = self.imgenc(img)
        x_v = self.visual_mlp(img)
        x = self.gmu(x_v, x_t)
        
        return self.logistic_mlp(x)
