#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import argparse
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME

from mmbt.data.helpers import get_data_loaders
from mmbt.models import get_model
from mmbt.utils.logger import create_logger
from mmbt.utils.utils import *
from mmbt.models.vilbert import BertConfig

from os.path import expanduser
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=128)
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased"])
    parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")
    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--embed_sz", type=int, default=300)
    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)
    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="bow", choices=["bow", "img", "bert", "concatbow", "concatbow16", "concatbert", "mmbt", "gmu", "mmtr", "mmbtp", "mmdbt", "vilbert", "mmbt3"])
    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--name", type=str, default="nameless")
    parser.add_argument("--num_image_embeds", type=int, default=1)
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/path/to/save_dir/")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101"])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)
    parser.add_argument('--output_gates', action='store_true', help='Store GMU gates of test dataset to a file (default: false)')
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "att", "cls_att", "vert_att"], help='Type of pooling technique for BERT models')
    
    '''AdaptaBERT parameter'''
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned (with the cloze-style LM objective) BERT model?")
    
    '''MMTransformer parameters'''
    parser.add_argument('--vonly', action='store_false', help='use the crossmodal fusion into v (default: False)')
    parser.add_argument('--lonly', action='store_false', help='use the crossmodal fusion into l (default: False)')
    parser.add_argument("--orig_d_v", type=int, default=2048)
    parser.add_argument("--orig_d_l", type=int, default=768)
    parser.add_argument("--v_len", type=int, default=3)
    parser.add_argument("--l_len", type=int, default=512)
    parser.add_argument('--attn_dropout', type=float, default=0.1, help='attention dropout')
    parser.add_argument('--attn_dropout_v', type=float, default=0.0, help='attention dropout (for visual)')
    parser.add_argument('--relu_dropout', type=float, default=0.1, help='relu dropout')
    parser.add_argument('--embed_dropout', type=float, default=0.25, help='embedding dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1, help='residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0, help='output layer dropout')
    parser.add_argument('--nlevels', type=int, default=5, help='number of layers in the network (default: 5)')
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--num_heads', type=int, default=5, help='number of heads for the transformer network (default: 5)')
    parser.add_argument('--attn_mask', action='store_false', help='use attention mask for Transformer (default: true)')
        
    '''ViLBERT parameters'''
    parser.add_argument("--from_pretrained", type=str, default=expanduser("~")+"/vilbert-multi-task/save/multi_task_model.bin")
    parser.add_argument("--config_file", type=str, default=expanduser("~")+"/vilbert-multi-task/config/bert_base_6layer_6conect.json")
    parser.add_argument("--vision_scratch", action="store_true", help="whether pre-trained the image or not.")

def get_criterion(args):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt", "mmbtp", "mmbt3"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    elif args.model == "vilbert":
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        base_lr = args.lr
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if "vil_" in key:
                    lr = 1e-4
                else:
                    if args.vision_scratch:
                        if key[12:] in bert_weight_name:
                            lr = base_lr
                        else:
                            lr = 1e-4
                    else:
                        lr = base_lr
                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]
                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=base_lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )


def model_eval(i_epoch, data, model, args, criterion, store_preds=False, output_gates=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        all_gates = []  # For gmu gate interpretability
        for batch in data:
            if output_gates:
                loss, out, tgt, gates = model_forward(i_epoch, model, args, criterion, batch, output_gates)
            else:
                loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)
            if output_gates:
                gates = gates.cpu().detach().numpy()
                all_gates.append(gates)

    metrics = {"loss": np.mean(losses)}
    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)
    
    if store_preds:
        if output_gates:
            all_gates = np.vstack(all_gates)
            print("gates: ", all_gates.shape)
            store_preds_to_disk(tgts, preds, args, all_gates)
        else:
            store_preds_to_disk(tgts, preds, args)

    return metrics


def model_forward(i_epoch, model, args, criterion, batch, gmu_gate=False):
    if args.model == "mmbt3":
        txt, segment, mask, mm_mask, img, tgt = batch
    else:
        txt, segment, mask, img, tgt = batch

    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    if args.model == "bow":
        txt = txt.cuda()
        out = model(txt)
    elif args.model == "img":
        img = img.cuda()
        out = model(img)
    elif args.model in ["concatbow", "concatbow16", "gmu"]:
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.cuda(), mask.cuda(), segment.cuda()
        out = model(txt, mask, segment)
    elif args.model in ["concatbert", "mmtr"]:
        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        if gmu_gate:
            out, gates = model(txt, mask, segment, img, gmu_gate)
        else:
            out = model(txt, mask, segment, img)
    elif args.model == "vilbert":
        txt, img = txt.cuda(), img.cuda()
        out = model(txt, img)
    else:
        assert args.model in ["mmbt", "mmbtp", "mmdbt", "mmbt3"]
        for param in model.enc.img_encoder.parameters():
            param.requires_grad = not freeze_img
        for param in model.enc.encoder.parameters():
            param.requires_grad = not freeze_txt

        txt, img = txt.cuda(), img.cuda()
        mask, segment = mask.cuda(), segment.cuda()
        
        if args.model == "mmbt3":
            mm_mask = mm_mask.cuda()
            out = model(txt, mask, mm_mask, img)
        else:
            out = model(txt, mask, segment, img)

    tgt = tgt.cuda()
    loss = criterion(out, tgt)
    
    if gmu_gate:
        return loss, out, tgt, gates
    else:
        return loss, out, tgt


def train(args):

    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, args.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader, test_loader = get_data_loaders(args)
    
    if args.trained_model_dir: # load in fine-tuned (with cloze-style LM objective) model
        args.previous_state_dict_dir = os.path.join(args.trained_model_dir, WEIGHTS_NAME)

    if args.model == "vilbert":
        config = BertConfig.from_json_file(args.config_file)
        model = get_model(args, config)
    else:
        model = get_model(args)

    criterion = get_criterion(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    logger.info(model)
    model.cuda()

    torch.save(args, os.path.join(args.savedir, "args.pt"))

    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.savedir, "checkpoint.pt")):
        checkpoint = torch.load(os.path.join(args.savedir, "checkpoint.pt"))
        start_epoch = checkpoint["epoch"]
        n_no_improve = checkpoint["n_no_improve"]
        best_metric = checkpoint["best_metric"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])

    logger.info("Training..")
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, _, _ = model_forward(i_epoch, model, args, criterion, batch)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics = model_eval(i_epoch, val_loader, model, args, criterion)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "model_best.pt"))
    model.eval()

    test_metrics = model_eval(
        np.inf, test_loader, model, args, criterion, store_preds=True, output_gates=args.output_gates
    )
    log_metrics(f"Test - ", test_metrics, args, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    cli_main()
