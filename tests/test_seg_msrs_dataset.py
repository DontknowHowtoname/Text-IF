# tests/test_seg_msrs_dataset.py
import os
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1] / "dataset" / "MSRS-main"


def test_alignment_counts():
    from seg_msrs.dataset import MSRSSegDataset
    train = MSRSSegDataset(ROOT, split="train")
    test = MSRSSegDataset(ROOT, split="test")
    assert len(train) == 1083
    assert len(test) == 361
    assert train.missing == [] and test.missing == []


def test_getitem_shapes_and_label_range():
    from seg_msrs.dataset import MSRSSegDataset
    ds = MSRSSegDataset(ROOT, split="test", train=False)
    img, label = ds[0]
    assert img.shape == (3, 480, 640)
    assert label.shape == (480, 640)
    assert label.dtype == torch.int64
    assert (label >= 0).all() and (label <= 8).all()


def test_train_crop_shape():
    from seg_msrs.dataset import MSRSSegDataset
    ds = MSRSSegDataset(ROOT, split="train", train=True, crop_size=480)
    img, label = ds[0]
    assert img.shape == (3, 480, 480)
    assert label.shape == (480, 480)


def test_train_aug_uses_nearest_label_and_ignore_pad():
    from seg_msrs.dataset import MSRSSegDataset
    # 00002D.png's label has NON-CONTIGUOUS uniques {0, 1, 2, 4}: a BILINEAR
    # label resize would interpolate id 3 at class boundaries, so the subset
    # assertion below actually catches a NEAREST -> BILINEAR regression.
    # (00001D.png has contiguous {0, 1, 2, 3} and cannot catch it.)
    ref_name = "00002D.png"
    ds = MSRSSegDataset(ROOT, split="train", train=True,
                        scale_range=(0.5, 0.5), crop_size=480)
    idx = next(i for i, (_, lbl) in enumerate(ds.items)
               if os.path.basename(lbl) == ref_name)
    orig_uniques = set(torch.unique(
        torch.as_tensor(np.array(Image.open(ds.items[idx][1])))).tolist())
    assert len(orig_uniques) >= 3, "reference label must be multi-class for this check"
    assert orig_uniques != set(range(len(orig_uniques))), \
        "reference label ids must be non-contiguous for this check"

    random.seed(0)
    for _ in range(5):
        img, label = ds[idx]
        uniques = torch.unique(label)
        # downscale + pad with ignore_index must introduce the 255 padding
        assert 255 in uniques
        # all non-255 ids stay in the valid class range
        valid = uniques[uniques != 255]
        assert (valid >= 0).all() and (valid <= 8).all()
        # NEAREST resize keeps ids within the original label's value set;
        # BILINEAR would interpolate ids outside orig_uniques
        assert set(uniques.tolist()) <= (orig_uniques | {255})


def test_dataloader_windows_spawn_smoke():
    from torch.utils.data import DataLoader
    from seg_msrs.dataset import MSRSSegDataset
    ds = MSRSSegDataset(ROOT, split="test", train=False)
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    for img, label in loader:
        assert img.shape == (img.shape[0], 3, 480, 640)
        assert label.shape == (label.shape[0], 480, 640)
        break
    batches = 0
    for img, label in loader:
        batches += 1
        if batches == 2:
            break
    assert batches == 2
