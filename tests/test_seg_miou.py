# tests/test_seg_miou.py
import numpy as np
import torch


def test_confusion_to_iou_perfect_and_partial():
    from seg_msrs.eval_seg_miou import confusion_to_iou
    conf = np.array([[10, 0], [2, 8]], dtype=np.int64)  # 2 classes
    iou = confusion_to_iou(conf)
    # class 0: TP=10, the 2 (conf[1,0]) is FP for class 0 -> union = 10+2
    assert abs(iou[0] - 10 / 12) < 1e-9
    assert abs(iou[1] - 8 / 10) < 1e-9


def test_update_confusion_shapes():
    from seg_msrs.eval_seg_miou import ConfusionMeter
    m = ConfusionMeter(3, ignore_index=255)
    pred = torch.tensor([[0, 1, 255], [2, 2, 1]])
    label = torch.tensor([[0, 1, 1], [2, 0, 255]])
    m.update(pred, label)
    conf = m.conf
    # valid pixels (pred AND label != 255): (0,0),(1,1),(2,2),(2,0);
    # the pixel with pred=1, label=255 (flat idx 5) is masked out
    assert conf[0, 0] == 1 and conf[1, 1] == 1 and conf[2, 2] == 1 and conf[2, 0] == 1
