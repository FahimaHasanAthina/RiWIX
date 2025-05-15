<<<<<<< HEAD
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
=======
import torch
import argparse
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from swin_unet import swin_Unet
from torch.nn.modules.loss import CrossEntropyLoss
from utils import DiceLoss
from tqdm import tqdm
from utils import (
    load_checkpoint,
    save_checkpoint,
    check_accuracy,
    save_predictions_as_imgs,
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.mask_files = os.listdir(mask_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        mask_name = img_name[:-4]+'.png'
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        mask = transforms.ToTensor()(mask) 

        if self.transform:
            image = self.transform(image)

        return image, mask

# Define transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)        


def train(args, model):

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    root_path = args.root_path


    # Create dataset and dataloader
    traindataset = SegmentationDataset(os.path.join(root_path, 'train', 'img'), os.path.join(args.root_path, 'train', 'label'), transform=data_transforms)
    print(len(traindataset))
    train_dataloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    valdataset = SegmentationDataset(os.path.join(root_path, 'val', 'img'), os.path.join(args.root_path, 'val', 'label'), transform=data_transforms)
    print(len(valdataset))
    val_dataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_dataloader) 
    print(f'{len(train_dataloader)} iterations per epoch. {max_iterations} max iterations')
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = 10e10

    for epoch_num in iterator:
        model.train()
        batch_dice_loss = 0
        batch_ce_loss = 0
        for i_batch, sampled_batch in tqdm(enumerate(train_dataloader), desc=f"Train: {epoch_num}", total=len(train_dataloader), leave=False):
            image_batch, label_batch = sampled_batch[0], sampled_batch[1]
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch.squeeze(1).long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.4 * loss_ce + 0.6 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1

            batch_dice_loss += loss_dice.item()
            batch_ce_loss += loss_ce.item()

        batch_ce_loss /= len(train_dataloader)
        batch_dice_loss /= len(train_dataloader)
        batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
        print(f'Train epoch: {epoch_num}, loss : {batch_loss}, loss_ce: {batch_ce_loss}, loss_dice: {batch_dice_loss}')
        if (epoch_num + 1) % args.eval_interval == 0:
            model.eval()
            batch_dice_loss = 0
            batch_ce_loss = 0
            with torch.no_grad():
                for i_batch, sampled_batch in tqdm(enumerate(val_dataloader), desc=f"Val: {epoch_num}",
                                                   total=len(val_dataloader), leave=False):
                    image_batch, label_batch = sampled_batch[0], sampled_batch[1]
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss_ce = ce_loss(outputs, label_batch.squeeze(1).long())
                    loss_dice = dice_loss(outputs, label_batch, softmax=True)
                    batch_dice_loss += loss_dice.item()
                    batch_ce_loss += loss_ce.item()

                batch_ce_loss /= len(val_dataloader)
                batch_dice_loss /= len(val_dataloader)
                batch_loss = 0.4 * batch_ce_loss + 0.6 * batch_dice_loss
                print(f'Val epoch: {epoch_num}, loss : {batch_loss}, loss_ce: {batch_ce_loss}, loss_dice: {batch_dice_loss}')
                if batch_loss < best_loss:
                    # save model
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer":optimizer.state_dict(),
                    }
                    save_checkpoint(checkpoint, "new_checkpoint.pth.tar")
                    # print some examples to a folder
                    save_predictions_as_imgs(
                        val_dataloader, model, folder=args.output_dir, device=device
                    )
                    best_loss = batch_loss
                else:
                    save_mode_path = os.path.join(args.model_dir, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
        

    return "Training Finished!"


parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../data/Synapse/train_npz', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')

parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--patch_size', type=int, default=4, help='patch size')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--resume', type=bool, help='resume from checkpoint')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument("--eval_interval", default=1, type=int)
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--model_dir", type=str, default="C:\\Users\\fa578s\\Desktop\\Automatic-river-width-calculator-main\\", help="save model directory" )

args = parser.parse_args()
if args.dataset == "GLH":
    args.root_path = os.path.join(args.root_path, "GLH")

if __name__ == "__main__":

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    model = swin_Unet(img_size=args.img_size, num_classes=args.num_classes).cuda()
    if args.resume == True:
        load_checkpoint(torch.load("new_checkpoint.pth.tar"), model)

    train(args, model)










# for name, param in model.named_parameters():
#     print(name, param.shape)
# print(model)

# def test():
#     x = torch.randn((1, 3, 224, 224))
#     model = swin_Unet()
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape==x.shape

# test()









# for name, param in model.named_parameters():
#     print(name, param.shape)
# print(model)

# def test():
#     x = torch.randn((1, 3, 224, 224))
#     model = swin_Unet()
#     preds = model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape==x.shape

# test()
>>>>>>> 768dfd014359f903f45a38782239f7452c9f8085
