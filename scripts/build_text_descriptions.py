"""Offline attribute extraction + text composition for FLIR-align-3class.

Run as a script to build per-image attribute cache (attrs.json) for train/test:
    python scripts/build_text_descriptions.py

Imported by FLIRPromptDataSet to compose text online during training.
"""
import os
import json
import statistics
from collections import Counter

# Class mapping (user-confirmed)
CLASS_NAMES = {0: 'people', 1: 'car', 2: 'bicycle'}
CLASS_PLURAL = {'people': 'people', 'car': 'cars', 'bicycle': 'bicycles'}

# Per-class distance thresholds (data-driven, spec section 3.2)
# far < p25, p25 <= mid < p75, near >= p75
DISTANCE_THRESHOLDS = {
    0: {'far': 0.0008, 'mid': 0.0040},   # people
    1: {'far': 0.0010, 'mid': 0.0083},   # car
    2: {'far': 0.0011, 'mid': 0.0043},   # bicycle
}


def extract_attributes(label_path, image_filename):
    """Extract 6 attributes from a YOLO label file.

    Args:
        label_path: path to the .txt label file (YOLO format)
        image_filename: filename of the IR image (used for time_of_day)

    Returns:
        dict with keys: main_class, main_class_name, distance, count,
                        time_of_day, co_occurrence, position,
                        median_area, n_main
        or None if label is empty.
    """
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return None

    bboxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        x, y, w, h = map(float, parts[1:5])
        bboxes.append({
            'class': cls, 'x': x, 'y': y, 'w': w, 'h': h,
            'area': w * h,
        })

    if not bboxes:
        return None

    # Count per class
    cls_counter = Counter(b['class'] for b in bboxes)
    main_class = cls_counter.most_common(1)[0][0]
    main_bboxes = [b for b in bboxes if b['class'] == main_class]

    # Distance from median area of main class
    median_area = statistics.median(b['area'] for b in main_bboxes)
    thresh = DISTANCE_THRESHOLDS[main_class]
    if median_area < thresh['far']:
        distance = 'far'
    elif median_area < thresh['mid']:
        distance = 'mid'
    else:
        distance = 'near'

    # Count
    n_main = len(main_bboxes)
    if n_main == 1:
        count = 'single'
    elif n_main <= 3:
        count = 'few'
    else:
        count = 'crowd'

    # Time of day from filename
    fname_lower = image_filename.lower()
    if '_day' in fname_lower:
        time_of_day = 'day'
    elif '_night' in fname_lower:
        time_of_day = 'night'
    else:
        time_of_day = 'unknown'

    # Co-occurrence
    classes_present = set(cls_counter.keys())
    if classes_present == {main_class}:
        co_occurrence = 'alone'
    elif classes_present == {0, 1, 2}:
        co_occurrence = 'mixed_traffic'
    elif len(classes_present) >= 2:
        other_classes = classes_present - {main_class}
        other_names = sorted(CLASS_NAMES[c] for c in other_classes)
        co_occurrence = 'with_' + '_and_'.join(other_names)
    else:
        co_occurrence = 'alone'

    # Position from mean x_center of main class
    mean_x = statistics.mean(b['x'] for b in main_bboxes)
    if mean_x < 0.33:
        position = 'left_side'
    elif mean_x < 0.67:
        position = 'center'
    else:
        position = 'right_side'

    return {
        'main_class': main_class,
        'main_class_name': CLASS_NAMES[main_class],
        'distance': distance,
        'count': count,
        'time_of_day': time_of_day,
        'co_occurrence': co_occurrence,
        'position': position,
        'median_area': median_area,
        'n_main': n_main,
    }


# ------------------------- Text composition -------------------------
# Defined in Task 4 below.
