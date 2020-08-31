#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from mmbt.models.bert import BertClf
from mmbt.models.bow import GloveBowClf
from mmbt.models.concat_bert import MultimodalConcatBertClf
from mmbt.models.concat_bow import  MultimodalConcatBowClf, MultimodalConcatBow16Clf, MLPGenreClf
from mmbt.models.image import ImageClf
from mmbt.models.mmbt import MultimodalBertClf
from mmbt.models.gmu import GMUClf
from mmbt.models.mmtr import MMTransformerClf
from mmbt.models.mmbtp import MultimodalBertTransfClf
from mmbt.models.mmdbt import MultimodalDistilBertClf
from mmbt.models.vilbert import VILBertForVLTasks


MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbow16": MLPGenreClf,
    "gmu": GMUClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    "mmtr": MMTransformerClf,
    "mmbtp": MultimodalBertTransfClf,
    "mmdbt": MultimodalDistilBertClf,
    "vilbert": VILBertForVLTasks,
}


def get_model(args, config=None):
    if config:
        #config.num_image_embeds = args.num_image_embeds
        config.args = args
        return VILBertForVLTasks.from_pretrained(args.from_pretrained,
                                                 config=config)
    return MODELS[args.model](args)