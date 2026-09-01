import csv
import json
from pathlib import Path

import pytest


def test_read_resume_state(tmp_path):
    from seg_msrs.train_segformer_msrs import read_resume_state
    # no csv -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        read_resume_state(tmp_path)
    with open(tmp_path / "train_log.csv", "w", newline="") as f:
        csv.writer(f).writerow([130, 0.04, 0.7258, "{}"])
    json.dump({"mIoU": 0.7278, "per_class": {}}, open(tmp_path / "best_miou.json", "w"))
    start_epoch, best = read_resume_state(tmp_path)
    assert start_epoch == 131
    assert abs(best - 0.7278) < 1e-9


def test_read_resume_state_no_best_json(tmp_path):
    from seg_msrs.train_segformer_msrs import read_resume_state
    with open(tmp_path / "train_log.csv", "w", newline="") as f:
        csv.writer(f).writerow([10, 0.1, 0.5, "{}"])
    start_epoch, best = read_resume_state(tmp_path)
    assert start_epoch == 11
    assert best == -1.0
