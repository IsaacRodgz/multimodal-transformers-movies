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
from mmbt.models.mmbt3 import MultimodalBertThreeClf
from mmbt.models.gmu import GMUClf
from mmbt.models.mmtr import MMTransformerClf, MMTransformerGMUClf, MMTransformerUniClf, TransformerClf, MMTransformerUniBi
from mmbt.models.mmbtp import MultimodalBertTransfClf
from mmbt.models.mmdbt import MultimodalDistilBertClf
from mmbt.models.vilbert import VILBertForVLTasks
from mmbt.models.mmbt_rating import MultimodalBertRatingClf
from mmbt.models.mmtr_rating import MMTransformerRatingClf


MODELS = {
    "bert": BertClf,
    "bow": GloveBowClf,
    "concatbow": MultimodalConcatBowClf,
    "concatbow16": MLPGenreClf,
    "gmu": GMUClf,
    "concatbert": MultimodalConcatBertClf,
    "img": ImageClf,
    "mmbt": MultimodalBertClf,
    "mmbt3": MultimodalBertThreeClf,
    "mmtr": MMTransformerUniBi,
    "mmbtp": MultimodalBertTransfClf,
    "mmdbt": MultimodalDistilBertClf,
    "vilbert": VILBertForVLTasks,
    "mmbtrating": MultimodalBertRatingClf,
    "mmtrrating": MMTransformerRatingClf,
}


def get_model(args, config=None):
    # Initialize ViLBERT model
    if config:
        config.args = args
        return VILBertForVLTasks.from_pretrained(args.from_pretrained,
                                                 config=config)
    # Initialize all other models
    return MODELS[args.model](args)