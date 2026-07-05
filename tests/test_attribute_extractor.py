"""Tests for attribute extraction from FLIR YOLO labels."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import os
from scripts.build_text_descriptions import extract_attributes, CLASS_NAMES

DATA_ROOT = 'D:/StudyFiles/MachineLearning/datasets/FLIR-align-3class/FLIR-align-3class'


def _find_sample_with_class(target_cls):
    """Find a real label file where target_cls is the majority (main) class."""
    from collections import Counter
    label_dir = os.path.join(DATA_ROOT, 'labels', 'train')
    for fn in sorted(os.listdir(label_dir))[:2000]:
        path = os.path.join(label_dir, fn)
        with open(path) as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            continue
        classes = [int(l.split()[0]) for l in lines]
        if classes and Counter(classes).most_common(1)[0][0] == target_cls:
            return path, fn
    return None, None


def test_extract_on_real_label_person():
    """Attribute extraction on a real label file containing people."""
    path, fname = _find_sample_with_class(0)
    assert path is not None, "No sample with class 0 found"
    attrs = extract_attributes(path, fname)
    assert attrs is not None
    assert attrs['main_class'] == 0
    assert attrs['main_class_name'] == 'people'
    assert attrs['distance'] in {'far', 'mid', 'near'}
    assert attrs['count'] in {'single', 'few', 'crowd'}
    assert attrs['time_of_day'] in {'day', 'night', 'unknown'}
    assert attrs['position'] in {'left_side', 'center', 'right_side'}
    assert isinstance(attrs['co_occurrence'], str)


def test_extract_on_real_label_car():
    """Attribute extraction on a real label file containing cars."""
    path, fname = _find_sample_with_class(1)
    assert path is not None
    attrs = extract_attributes(path, fname)
    assert attrs is not None
    assert attrs['main_class_name'] in {'people', 'car', 'bicycle'}


def test_time_of_day_from_filename():
    """Filename _day should produce time_of_day='day'."""
    path, fname = _find_sample_with_class(0)
    attrs = extract_attributes(path, fname)
    if '_day' in fname.lower():
        assert attrs['time_of_day'] == 'day'
    elif '_night' in fname.lower():
        assert attrs['time_of_day'] == 'night'


def test_empty_label_returns_none():
    """Empty label file should return None (use fallback template)."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('')
        path = f.name
    try:
        attrs = extract_attributes(path, 'dummy_day.jpeg')
        assert attrs is None
    finally:
        os.unlink(path)


def test_single_object_is_single_count():
    """One bbox in label file should produce count='single'."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        f.write('0 0.5 0.5 0.1 0.1\n')  # 1 person, mid-area
        path = f.name
    try:
        attrs = extract_attributes(path, 'dummy_day.jpeg')
        assert attrs['count'] == 'single'
        assert attrs['main_class'] == 0
    finally:
        os.unlink(path)


def test_many_objects_is_crowd_count():
    """5 bboxes of same class should produce count='crowd'."""
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False) as f:
        for i in range(5):
            f.write(f'1 0.{i:02d} 0.5 0.05 0.05\n')  # 5 cars
        path = f.name
    try:
        attrs = extract_attributes(path, 'dummy_day.jpeg')
        assert attrs['count'] == 'crowd'
        assert attrs['main_class'] == 1
    finally:
        os.unlink(path)


if __name__ == '__main__':
    test_extract_on_real_label_person()
    print("PASS: person attribute extraction")
    test_extract_on_real_label_car()
    print("PASS: car attribute extraction")
    test_time_of_day_from_filename()
    print("PASS: time_of_day from filename")
    test_empty_label_returns_none()
    print("PASS: empty label returns None")
    test_single_object_is_single_count()
    print("PASS: single object count")
    test_many_objects_is_crowd_count()
    print("PASS: crowd count")
    print("All attribute extractor tests passed.")
