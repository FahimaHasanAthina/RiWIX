
"""
datasets class
"""
import json
import os

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class GLHDataset(Dataset):
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



def build_dataset(is_train, args):
    """buld_dataset."""
    transform = build_transform(is_train, args)

    if args.dataset == "GLH":
        if is_train:
            image_dir = os.path.join(args.root_path, 'train', 'img')
            mask_dir = os.path.join(args.root_path, 'train', 'label')
        else:
            image_dir = os.path.join(args.root_path, 'val', 'img')
            mask_dir = os.path.join(args.root_path, 'val', 'label')
        dataset = GLHDataset(image_dir, mask_dir, transform)

    return dataset


def build_transform(is_train, args):
    """build transform."""
    resize_im = args.img_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.img_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] =\
                transforms.RandomCrop(args.img_size, padding=4)
        return transform

    t = []  # Test-time transformations.
    if resize_im:
        size = args.img_size
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.img_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
