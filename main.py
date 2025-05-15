import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from pathlib import Path
from swin_unet import swin_Unet
from torch.nn.modules.loss import CrossEntropyLoss, BCEWithLogitsLoss
from utils import DiceLoss
from timm.utils import ModelEma, NativeScaler, get_state_dict
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from engine import evaluate, train_one_epoch
import json
import utils
from datasets import build_dataset
import time
import datetime
import wandb


def get_args_parser():

    parser = argparse.ArgumentParser('train and evaluation script', add_help=False)

    # debug parameters
    parser.add_argument("--debug", action="store_true", help="enable debug mode")

    # basic training parameters
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epochs", default=300, type=int)
    parser.add_argument("--save_freq", default=10, type=int)

    #additional args
    parser.add_argument("--model_kwargs", type=str, default="{}", help="additional parameters for model")
    parser.add_argument("--disable_amp", action="store_true", default=False)

    # model parameters
    parser.add_argument("--model", default="new_checkpoint.pth.tar", type=str, metavar="MODEL", help="Name of model to train")
    parser.add_argument("--img_size", type=int, default=896, help='input patch size of network input')
    parser.add_argument("--patch_size", type=int, default=4, help='patch size')
    parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)")
    parser.add_argument("--drop-path", type=float, default=0.0, metavar="PCT", help="Drop path rate (default: 0.)")
    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=True)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument("--model-ema-force-cpu", action="store_true", default=False, help="")

    # optimizer parameters
    parser.add_argument("--opt", default="adamw", type=str, metavar="OPTIMIZER", help='Optimizer (default: "adamu)')
    parser.add_argument("--opt-eps", default=1e-8, type=float, metavar="EPSILON", help="Optimizer Epsilon (default: 1e-8)")
    parser.add_argument("--opt-betas", default=None, type=float, nargs="+", metavar="BETA", help="Optimizer Betas (default: None, use opt default)")
    parser.add_argument("--clip-grad", type=float, default=None, metavar="NORM", help="Clip gradient norm (default: None, no clipping)")
    parser.add_argument("--momentum", type=float, default=0.9, metavar="M", help="SGD momentum (default: 0.9)")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)")
    
    # Learning rate schedule parameters
    parser.add_argument("--sched", default="cosine", type=str, metavar="SCHEDULER", help='LR scheduler (default: "cosine"')
    parser.add_argument("--lr", type=float, default=5e-4, metavar="LR", help="learning rate (default: 5e-4)")
    parser.add_argument("--lr-noise", type=float, nargs="+", default=None, metavar="pct, pct", help="learning rate noise on/off epoch percentages")
    parser.add_argument("--lr-noise-pct", type=float, default=0.67, metavar="PERCENT", help="learning rate noise limit percent (default: 0.67)")
    parser.add_argument("--lr-noise-std", type=float, default=1.0, metavar="STDDEV", help="learning rate noise std-dev (default: 1.0)")
    parser.add_argument("--warmup-lr", type=float, default=1e-6, metavar="LR", help="warmup learning rate (default: 1e-6)")
    parser.add_argument("--min-lr", type=float, default=1e-5, metavar="LR", help="lower lr bound for cyclic schedulers that hit 0 (1e-5)")
    parser.add_argument("--decay-epochs", type=float, default=30, metavar="N", help="epoch interval to decay LR")
    parser.add_argument("--warmup-epochs", type=int, default=5, metavar="N", help="epochs to warmup LR, if scheduler supports")
    parser.add_argument("--cooldown-epochs", type=int, default=10, metavar="N", help="epochs to cooldown LR at min_lr, after cyclic schedule ends")
    parser.add_argument("--patience-epochs", type=int, default=10, metavar="N", help="patience epochs for Plateau LR scheduler (default: 10")
    parser.add_argument("--decay-rate", "--dr", type=float, default=0.1, metavar="RATE", help="LR decay rate (default: 0.1)")

    # Augmentation parameters
    parser.add_argument("--color-jitter", type=float, default=0.4, metavar="PCT", help="Color jitter factor (default: 0.4)")
    parser.add_argument("--aa", type=str, default="rand-m9-mstd0.5-inc1", metavar="NAME", help='Use AutoAugment policy. "v0" or "original". " + \
                    "(default: rand-m9-mstd0.5-inc1)')
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)")
    parser.add_argument("--train-interpolation", type=str, default="bicubic", help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument("--reprob", type=float, default=0.25, metavar="PCT", help="Random erase prob (default: 0.25)")
    parser.add_argument("--remode", type=str, default="pixel", help='Random erase mode (default: "pixel")')
    parser.add_argument("--recount", type=int, default=1, help="Random erase count (default: 1)")
    parser.add_argument("--resplit", action="store_true", default=False, help="Do not random erase first (clean) augmentation split")

    # Dataset parameters
    parser.add_argument("--root_path", default="C:\\Users\\fa578s\\Desktop\\preprocessed\\GLH\\", type=str, help="root dir for data")
    parser.add_argument("--dataset", default="GLH", choices=["GLH", "NAIP"], type=str, help="dataset path")
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes for segmentation')
    parser.add_argument('--num_channels', type=int, default=1, help='output channels of the network')
    parser.add_argument("--output_dir", default="D:\\river_width\\checkpoints\\", type=str, help="output dir")
    parser.add_argument("--device", default="cuda", help="device to use for ttraining / testing")
    parser.add_argument("--seed", type=int, default=1234, help='random seed')
    parser.add_argument("--resume", default=False, type=bool, help='resume from checkpoint')
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument("--eval_interval", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin-mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument("--world_size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist_url", default="env://", help="url used to set up distributed training")

    return parser

def main(args):
    """
    training main function
    """
    # Initialize wandb
    wandb.init(
        project="RiWiX",  # Project name on wandb
        name=f"exp_riwix_lr{args.lr}_bs{args.batch_size}",  # Run name with key params
        config=vars(args),  # Log args as config
    )
    utils.init_distributed_mode(args)

    print(args)
    # Debug mode.
    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=("0.0.0.0", 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)
    print(f'training samples: {len(dataset_train)}')
    print(f'validation samples: {len(dataset_val)}')

    # load dataset
    sampler_train = torch.utils.data.RandomSampler(dataset_train)

    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=int(1.5 * args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    print(f"Creating model: {args.model}")
    model = swin_Unet(
        img_size=args.img_size, 
        num_channels=args.num_channels, 
        patch_size=args.patch_size)
    
    print(model)

    model.to(device)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    optimizer = create_optimizer(args, model)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    # ce_loss = CrossEntropyLoss(label_smoothing=0.1)
    ce_loss = BCEWithLogitsLoss()
    dice_loss = DiceLoss(args.num_classes)

    output_dir = Path(args.output_dir)

    if args.resume == True:
        print("loading checkpoint................")
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint["state_dict"])
        model_without_ddp.load_state_dict(checkpoint["state_dict"])

        if (
                not args.eval
                and "optimizer" in checkpoint
                and "lr_scheduler" in checkpoint
                and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, args.num_classes, disable_amp=args.disable_amp)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%"
        )
        if args.output_dir and utils.is_main_process():
            with (output_dir / "test_log.txt").open("a") as f:
                log_stats = {
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "n_parameters": n_parameters,
                }
                f.write(json.dumps(log_stats) + "\n")
        return
    
    print("Start training")
    start_time = time.time()
    max_accuracy = 0.0

    # Initial checkpoint saving.
    if args.output_dir:
        checkpoint_paths = [output_dir / "checkpoint.pth"]
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master(
                {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": -1,  # Note: -1 means initial checkpoint.
                    "model_ema": get_state_dict(model_ema),
                    "args": args,
                },
                checkpoint_path,
            )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            ce_loss,
            dice_loss,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            disable_amp=args.disable_amp,
        )
        print(train_stats)

         # Log training stats to wandb
        wandb.log({
            "epoch": epoch,
            "learning_rate": train_stats["lr"],
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "train_dice_score": train_stats["dice_score"]
        })

        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            if epoch % args.save_freq == args.save_freq - 1 or epoch > args.epochs - 50:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": get_state_dict(model_ema),
                        "args": args,
                    },
                    checkpoint_path,
                )

        if epoch % args.save_freq == args.save_freq - 1 or epoch > args.epochs - 50:
            test_stats = evaluate(data_loader_val, model, device, args.num_classes, disable_amp=args.disable_amp)
            print(
                f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc']:.1f}%"
            )

            # Log validation stats to wandb
            wandb.log({
                "val_loss": test_stats["loss"],
                "val_acc": test_stats["acc"],
                "val_dice_score": test_stats["dice_score"]
            })

            max_accuracy = max(max_accuracy, test_stats["acc"])
            print(f"Max accuracy: {max_accuracy:.2f}%")

            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                **{f"test_{k}": v for k, v in test_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }
        else:
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "RiWiX training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)