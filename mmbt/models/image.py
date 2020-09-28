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
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures.image_list import ImageList


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

    
class ImageEncoderFasterRCNN(nn.Module):
    def __init__(self, args):
        super(ImageEncoderFasterRCNN, self).__init__()
        self.args = args
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 2048
        self.model = build_model(cfg)

    def forward(self, x):
        self.model.eval()
        images = ImageList.from_tensors([t for t in x], size_divisibility=self.model.backbone.size_divisibility).to("cuda")
        inputs = [{"image": img, "height": img.shape[0], "width": img.shape[1]} for img in images]
        features = self.model.backbone(images.tensor) # Get backbone features (p2, p3, ..., p6)
        proposals, _ = self.model.proposal_generator(images, features) # Get (at most) 1k proposed boxes
        features_ = [features[f] for f in self.model.roi_heads.box_in_features] # Extract features (p2, p3, ..., p6)
        pred_boxes = [x.proposal_boxes for x in proposals] # Extract all proposal bboxes
        box_features = self.model.roi_heads.box_pooler(features_, pred_boxes) # Extract RoIs (N*B, 256, 7, 7) where N is almost always 1000
        box_features = self.model.roi_heads.box_head(box_features) # Extract vector image features (N*B, 2048)
        predictions = self.model.roi_heads.box_predictor(box_features) # Get prediction tuple ((N*B, 81), (N*B, 320))
        pred_instances, pred_inds = self.model.roi_heads.box_predictor.inference(predictions, proposals) # Reduce to most likely 100 boxes at most with the corresponding indexes
        pred_instances = self.model.roi_heads.forward_with_given_boxes(features, pred_instances)
        pred_instances = self.model._postprocess(pred_instances, inputs, images.image_sizes)
        pred_boxes_lens = [len(p['instances'].pred_boxes) for p in pred_instances] # Get num of boxes per image feature obtained
        max_seq_len = min(pred_boxes_lens)
        selected_seq_len = min(max_seq_len, self.args.num_image_embeds) # Decide if there are enough RoIs to select self.args.num_image_embeds or cut to max_seq_len
        
        selected_box_features = [box_features[:self.args.num_image_embeds]]
        selected_bboxes = [pred_boxes[0][:self.args.num_image_embeds]]
        if True: # If no RoIs are left after processing, take the original RoIs
            proposal_lens = [len(p) for p in proposals]
            cum_idx = [sum(proposal_lens[:i]) for i in range(1, len(proposal_lens))]
            
            for i, idx in enumerate(cum_idx):
                selected_box_features.append(box_features[idx:idx+self.args.num_image_embeds])
                selected_bboxes.append(pred_boxes[i][:self.args.num_image_embeds])
            
            selected_box_features = torch.stack(selected_box_features, dim=0)
            bboxes = torch.stack([torch.cat((boxes.tensor,
                                             boxes.area().view(-1, 1)),
                                             dim=1
                                            ) for boxes in selected_bboxes], dim=0) # Extract selected_seq_len bboxes with additional area covered by bbox
            
            
        else:
            selected_box_features = torch.stack([box_features[pred_inds[i]][:selected_seq_len] for i in range(self.args.batch_sz)], dim=0) # Extract (B, selected_seq_len, 2048) features
            selected_bboxes = [x['instances'][:selected_seq_len].pred_boxes for x in pred_instances] # Extract selected_seq_len bboxes
            bboxes = torch.stack([torch.cat((boxes[:selected_seq_len].tensor,
                                             boxes[:selected_seq_len].area().view(-1, 1)),
                                             dim=1
                                            ) for boxes in selected_bboxes], dim=0) # Extract selected_seq_len bboxes with additional area covered by bbox
        
        return selected_box_features, bboxes


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
