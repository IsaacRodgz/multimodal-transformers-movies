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
import torchvision

from PIL import Image
import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg


class ImageEncoder(nn.Module):
    def __init__(self, args):
        super(ImageEncoder, self).__init__()
        self.args = args
        model = torchvision.models.resnet152(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)

        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )

        if args.num_image_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((args.num_image_embeds, 1))
        elif args.num_image_embeds == 4:
            self.pool = pool_func((2, 2))
        elif args.num_image_embeds == 6:
            self.pool = pool_func((3, 2))
        elif args.num_image_embeds == 8:
            self.pool = pool_func((4, 2))
        elif args.num_image_embeds == 9:
            self.pool = pool_func((3, 3))

    def forward(self, x):
        # Bx3x224x224 -> Bx2048x7x7 -> Bx2048xN -> BxNx2048
        out = self.pool(self.model(x))
        out = torch.flatten(out, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        return out  # BxNx2048


class ImageEncoder16(nn.Module):
    def __init__(self, args):
        super(ImageEncoder16, self).__init__()
        self.args = args
        self.model = torchvision.models.vgg16(pretrained=True)

    def forward(self, x):
        out = self.model.features(x)
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.model.classifier[0](out)
        
        return out


class ImageEncoderSeq(nn.Module):
    def __init__(self, args):
        super(ImageEncoderSeq, self).__init__()
        self.args = args
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        self.model = DefaultPredictor(cfg)

    def forward(self, x):
        
        outputs = predictor(x)
        
        all_features = []
        
        for pb in outputs["instances"].pred_boxes:
            
            bb = list(pb.detach().cpu().numpy())

            xi = int(bb[1])
            xf = int(bb[3])
            yi = int(bb[0])
            yf = int(bb[2])

            cropped_image = Image.fromarray(im2[xi:xf, yi:yf, ...].astype('uint8'), 'RGB')
            cropped_image = transform(cropped_image)
            all_features.append(cropped_image)
            
        all_features = torch.stack(all_features)


class ImageClf(nn.Module):
    def __init__(self, args):
        super(ImageClf, self).__init__()
        self.args = args
        self.img_encoder = ImageEncoder(args)
        self.clf = nn.Linear(args.img_hidden_sz * args.num_image_embeds, args.n_classes)

    def forward(self, x):
        x = self.img_encoder(x)
        x = torch.flatten(x, start_dim=1)
        out = self.clf(x)
        return out
