# seg_msrs/dataset.py
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_image(path, modality):
    img = Image.open(path)
    if modality == "ir":
        img = img.convert("L").convert("RGB")
    else:
        img = img.convert("RGB")
    return img  # PIL, HWC uint8


class MSRSSegDataset(Dataset):
    """MSRS vi/ir + Segmentation_labels. Filenames are shared across folders."""

    CLASSES = ("background", "car", "person", "bike", "curve",
               "car_stop", "guardrail", "color_cone", "bump")

    def __init__(self, root, split="train", modality="vi", train=False,
                 crop_size=480, scale_range=(0.5, 2.0), ignore_index=255):
        assert split in ("train", "test")
        assert modality in ("vi", "ir")
        self.root, self.split, self.modality = root, split, modality
        self.train, self.crop_size = train, crop_size
        self.scale_range, self.ignore_index = scale_range, ignore_index

        img_dir = os.path.join(root, split, modality)
        lbl_dir = os.path.join(root, split, "Segmentation_labels")
        names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".png"))
        self.items, self.missing = [], []
        for name in names:
            lbl = os.path.join(lbl_dir, name)
            if os.path.exists(lbl):
                self.items.append((os.path.join(img_dir, name), lbl))
            else:
                self.missing.append(name)

    def __len__(self):
        return len(self.items)

    def _transform(self, img, label):
        img = np.array(img, dtype=np.float32) / 255.0
        if self.train:
            # random scale
            sc = random.uniform(*self.scale_range)
            h, w = label.shape
            nh, nw = int(h * sc), int(w * sc)
            img = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize(
                (nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
            label = np.array(Image.fromarray(label).resize(
                (nw, nh), Image.NEAREST))
            # pad to crop size with ignore if smaller
            pad_h, pad_w = max(nh, self.crop_size), max(nw, self.crop_size)
            if pad_h > nh or pad_w > nw:
                img = np.pad(img, ((0, pad_h - nh), (0, pad_w - nw), (0, 0)))
                label = np.pad(label, ((0, pad_h - nh), (0, pad_w - nw)),
                               constant_values=self.ignore_index)
            # random crop
            top = random.randint(0, pad_h - self.crop_size)
            left = random.randint(0, pad_w - self.crop_size)
            img = img[top:top + self.crop_size, left:left + self.crop_size]
            label = label[top:top + self.crop_size, left:left + self.crop_size]
            # h-flip
            if random.random() < 0.5:
                img, label = img[:, ::-1], label[:, ::-1]
        img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        img = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return img, label

    def __getitem__(self, idx):
        img_path, lbl_path = self.items[idx]
        img = _load_image(img_path, self.modality)
        label = np.array(Image.open(lbl_path))
        return self._transform(img, label)


class ImageDirDataset(Dataset):
    """Inference-only dataset for arbitrary image dir (e.g. fused results)."""

    def __init__(self, image_dir):
        self.paths = sorted(
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".bmp", ".tif")))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        img = (img - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        return torch.from_numpy(img.transpose(2, 0, 1)).float(), os.path.basename(path)
