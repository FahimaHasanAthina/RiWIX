# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Train and eval functions used in main.py."""
import math
import sys
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from utils import DiceLoss
import utils
from utils import check_accuracy


def train_one_epoch(
        model: torch.nn.Module,
        ce_loss: torch.nn.Module,
        dice_loss: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        model_ema: Optional[ModelEma] = None,
        disable_amp: bool = False,
):
    """train one epoch function."""
    model.train()
    ce_loss.train()
    dice_loss.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(
        window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(
            data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # if mixup_fn is not None:
        #     samples, targets = mixup_fn(samples, targets)

        if disable_amp:
            # Disable AMP and try to solve the NaN issue.
            # Ref: https://github.com/facebookresearch/deit/issues/29
            outputs = model(samples)

            loss_ce = ce_loss(outputs, targets)
            loss_dice = dice_loss(outputs, targets, sigmoid=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
        else:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(samples)
                loss_ce = ce_loss(outputs, targets)
                loss_dice = dice_loss(outputs, targets, sigmoid=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        if disable_amp:
            loss.backward()
            optimizer.step()
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                    hasattr(optimizer, "is_second_order") and
                    optimizer.is_second_order
            )
            loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
            )

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        # B, C, H, W = outputs.shape
        # logits_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)
        # target_flat = targets.permute(0, 2, 3, 1).reshape(-1, C)
        # acc1, acc5 = accuracy(logits_flat, target_flat, topk=(1, 5))
        acc, dice_score = check_accuracy(outputs, targets)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(acc=acc)
        metric_logger.update(dice_score=dice_score)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, num_classes, disable_amp):
    """evaluation function."""
    ce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(num_classes)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if disable_amp:
            output = model(images)
            loss_ce = ce_loss(output, target)
            loss_dice = dice_loss(output, target, sigmoid=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice

        else:
            with torch.amp.autocast(device_type="cuda"):
                output = model(images)
                loss_ce = ce_loss(output, target)
                loss_dice = dice_loss(output, target, sigmoid=True)
                loss = 0.4 * loss_ce + 0.6 * loss_dice

        # Reshape logits and targets for accuracy
        # B, C, H, W = outputs.shape
        # logits_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)
        # target_flat = targets.permute(0, 2, 3, 1).reshape(-1, C)
        # acc1, acc5 = accuracy(logits_flat, target_flat, topk=(1, 5))

        acc, dice_score = check_accuracy(output, target)

        batch_size = images.shape[0] 
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc=acc)
        metric_logger.update(dice_score=dice_score)

    print('* Acc {acc.global_avg:.3f} DicdScore {dice.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(acc=metric_logger.acc, dice=metric_logger.dice_score, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

