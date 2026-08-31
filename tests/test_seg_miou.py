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


def test_update_out_of_range_raises():
    import pytest
    from seg_msrs.eval_seg_miou import ConfusionMeter
    m = ConfusionMeter(3, ignore_index=255)
    # label value 5 lands outside [0, 3) -> must raise, not mis-bucket
    with pytest.raises(ValueError, match="outside"):
        m.update(torch.tensor([[0, 1]]), torch.tensor([[0, 5]]))
    # out-of-range pred also raises
    with pytest.raises(ValueError, match="outside"):
        m.update(torch.tensor([[3, 1]]), torch.tensor([[0, 1]]))
    # a failed update must not pollute the confusion matrix
    assert m.conf.sum() == 0


def test_update_accumulates_across_calls():
    from seg_msrs.eval_seg_miou import ConfusionMeter
    m = ConfusionMeter(2)
    m.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    m.update(torch.tensor([[1, 0]]), torch.tensor([[1, 1]]))
    assert m.conf[0, 0] == 1 and m.conf[1, 1] == 2 and m.conf[0, 1] == 1
    assert m.conf.sum() == 4


def test_summarize_all_nan_class_is_none():
    from seg_msrs.eval_seg_miou import _summarize
    # class 1 row/col empty -> NaN IoU -> None in per_class
    conf = np.zeros((9, 9), dtype=np.int64)
    conf[0, 0] = 10
    conf[2, 2] = 5
    miou, per_class = _summarize(conf)
    assert per_class["background"] == 1.0
    assert per_class["car"] is None
    assert per_class["person"] == 1.0
    for c in ("bike", "curve", "car_stop", "guardrail", "color_cone", "bump"):
        assert per_class[c] is None
    # mean over the two present classes
    assert abs(miou - 1.0) < 1e-9
