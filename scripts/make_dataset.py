import json
import re
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict, Counter
from tempfile import TemporaryFile
import math
from numpy import asarray
from numpy import save
from numpy import load
from tqdm import tqdm

import detectron2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def resize_and_crop_image(input_file, output_box=[224, 224], fit=True):
        # https://github.com/BVLC/caffe/blob/master/tools/extra/resize_and_crop_images.py
        '''Downsample the image.
        '''
        if type(input_file) == str:
            img = Image.open(input_file)
        else:
            img = input_file
        #img.save("orig_"+input_file.split('/')[-1])
        box = output_box
        # preresize image with factor 2, 4, 8 and fast algorithm
        factor = 1
        while img.size[0] / factor > 2 * box[0] and img.size[1] * 2 / factor > 2 * box[1]:
            factor *= 2
        if factor > 1:
            img.thumbnail(
                (img.size[0] / factor, img.size[1] / factor), Image.NEAREST)

        # calculate the cropping box and get the cropped part
        if fit:
            x1 = y1 = 0
            x2, y2 = img.size
            wRatio = 1.0 * x2 / box[0]
            hRatio = 1.0 * y2 / box[1]
            if hRatio > wRatio:
                y1 = int(y2 / 2 - box[1] * wRatio / 2)
                y2 = int(y2 / 2 + box[1] * wRatio / 2)
            else:
                x1 = int(x2 / 2 - box[0] * hRatio / 2)
                x2 = int(x2 / 2 + box[0] * hRatio / 2)
            img = img.crop((x1, y1, x2, y2))

        # Resize the image with best quality algorithm ANTI-ALIAS
        img = img.resize(box, Image.ANTIALIAS).convert('RGB')
        #img = numpy.asarray(img, dtype='float32')
        return img
    

def get_image_feature(feature_extractor, image):
    with torch.no_grad():
        feature_images = feature_extractor.features(image)
        feature_images = feature_extractor.avgpool(feature_images)
        feature_images = torch.flatten(feature_images, 1)
        feature_images = feature_extractor.classifier[0](feature_images)
    
    return feature_images


def extract_visual(img_name):
    img = resize_and_crop_image(f"/home/est_posgrado_isaac.bribiesca/mmimdb/dataset/{img_name}.jpeg", (256,256))
    img = preprocess(img)
    img = img.unsqueeze(0)
    feature = get_image_feature(feature_extractor, img)
    
    return feature


preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
cfg.MODEL.WEIGHTS = "/home/est_posgrado_isaac.bribiesca/mmbt_experiments/model_final_f6e8b1.pkl"
predictor = DefaultPredictor(cfg)
feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)


data = [json.loads(l) for l in open('/home/est_posgrado_isaac.bribiesca/mmimdb/dev.jsonl')]

status = tqdm(data, total=int(len(data)))

for i, row in enumerate(status):
    path = '/home/est_posgrado_isaac.bribiesca'+row['img'][2:]
    img = Image.open(path)
    img = np.array(img).astype(np.float32)
    
    if len(img.shape) < 3:
        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
    elif img.shape[2] == 4:
        img = img[..., :3]
    
    status.set_postfix({'Shape': img.shape})
    
    outputs = predictor(img)
    
    all_features = [img]
    for pb in outputs["instances"].pred_boxes:

        bb = list(pb.cpu().numpy())

        xi = int(bb[1])
        xf = int(bb[3])
        yi = int(bb[0])
        yf = int(bb[2])
        
        #img_region = Image.fromarray(img[xi:xf, yi:yf, ...].astype('uint8'), 'RGB')
        #img_region = resize_and_crop_image(img_region, (256,256))
        #img_region = preprocess(img_region)
        #img_region = img_region.unsqueeze(0)
        #feature = get_image_feature(feature_extractor, img_region)
        #all_features.append(feature.squeeze(0))
        img_region = img[xi:xf, yi:yf, ...]
        all_features.append(img_region)
        
    img_name = row['img'].split('/')[-1][:-5]
    '''
    if len(all_features) > 0:
        img_feature = extract_visual(img_name).squeeze(0)
        all_features = torch.stack([img_feature]+all_features)
    else:
        all_features = extract_visual(img_name)
    '''
    #torch.save(all_features, f'../mmimdb/dataset_img/{img_name}.pt')
    np.savez(f'../../mmimdb/dataset_img_raw/{img_name}', *all_features)