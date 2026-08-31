# tests/test_seg_msrs_dataset.py
from pathlib import Path

import numpy as np
import pytest
import torch

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
