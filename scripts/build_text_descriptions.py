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


# ===================== Text composition =====================

# Per-slot phrase pools. Each attribute has a dict of value -> list of phrases.
TEXT_POOLS = {
    'count': {
        'single':  ['a single', 'one', 'a lone', 'a solitary'],
        'few':     ['a few', 'several', 'a small group of', 'two or three'],
        'crowd':   ['a crowd of', 'many', 'a large group of', 'numerous'],
    },
    'class': {
        'people':   ['pedestrian', 'person', 'individual', 'human subject'],
        'car':      ['car', 'vehicle', 'automobile'],
        'bicycle':  ['bicycle', 'bike', 'cyclist'],
    },
    'distance': {
        'far': ['far away', 'in the distance', 'at a far range'],
        'mid': ['at medium distance', 'mid-range', 'at a moderate distance'],
        'near':['up close', 'nearby', 'at close range'],
    },
    'position': {
        'left_side':  ['on the left side', 'on the left', 'to the left'],
        'center':     ['in the center', 'in the middle', 'straight ahead'],
        'right_side': ['on the right side', 'on the right', 'to the right'],
    },
    'time_of_day': {
        'day':     ['during daytime', 'under daylight', 'in daylight'],
        'night':   ['at night', 'under nighttime lighting', 'in low light'],
        'unknown': ['in unspecified lighting', 'under mixed lighting'],
    },
}

# Static fallback templates (per class) for robustness.
FALLBACK_TEMPLATES = {
    0: ['A pedestrian in the scene.',
        'Infrared-visible fusion emphasizing a person.',
        'A human subject in the field of view.',
        'An individual captured by thermal and visible cameras.',
        'A person in the urban environment.'],
    1: ['A vehicle in the scene.',
        'Infrared-visible fusion emphasizing a car.',
        'An automobile in the field of view.',
        'A car captured by thermal and visible cameras.',
        'A vehicle in the urban environment.'],
    2: ['A bicycle in the scene.',
        'Infrared-visible fusion emphasizing a bike.',
        'A cyclist in the field of view.',
        'A bicycle captured by thermal and visible cameras.',
        'A bike in the urban environment.'],
}


def compose_sentence(attrs, rng=None):
    """Compose an English sentence from 6 attributes.

    Args:
        attrs: dict from extract_attributes()
        rng: random.Random instance (default: module random)

    Returns:
        str: the composed sentence.
    """
    if rng is None:
        import random
        rng = random

    count_phrase = rng.choice(TEXT_POOLS['count'][attrs['count']])
    class_phrase = rng.choice(TEXT_POOLS['class'][attrs['main_class_name']])
    distance_phrase = rng.choice(TEXT_POOLS['distance'][attrs['distance']])
    position_phrase = rng.choice(TEXT_POOLS['position'][attrs['position']])

    # Resolve unknown time by random sampling (spec section 3.4)
    tod = attrs['time_of_day']
    if tod == 'unknown':
        tod = rng.choice(['day', 'night'])
    time_phrase = rng.choice(TEXT_POOLS['time_of_day'][tod])

    # Co-occurrence phrasing
    co = attrs['co_occurrence']
    if co == 'alone':
        co_phrase = rng.choice(['alone in the scene',
                                'with no other traffic',
                                'isolated from other objects'])
    elif co == 'mixed_traffic':
        co_phrase = rng.choice(['in mixed traffic',
                                'among people, cars and bicycles',
                                'surrounded by various road users'])
    else:
        # Parse 'with_X' or 'with_X_and_Y'
        suffix = co[len('with_'):]
        others = suffix.split('_and_')
        # Pluralize each
        plurals = [CLASS_PLURAL.get(o, o + 's') for o in others]
        co_phrase = f"alongside {' and '.join(plurals)}"

    return (f"{count_phrase} {class_phrase} {distance_phrase}, "
            f"{position_phrase}, {time_phrase}, {co_phrase}.")


def maybe_fallback(attrs, prob=0.075, rng=None):
    """With probability `prob`, return a static fallback template.

    Returns:
        str or None: a fallback sentence, or None if not triggered.
    """
    if rng is None:
        import random
        rng = random
    if rng.random() < prob:
        return rng.choice(FALLBACK_TEMPLATES[attrs['main_class']])
    return None


# ===================== CLI: rebuild attribute cache =====================

def _build_cache(split, data_root):
    """Scan a split's label folder and write attrs.json next to it."""
    label_dir = os.path.join(data_root, 'labels', split)
    ir_dir = os.path.join(data_root, 'infrared', split)
    out = {}
    files = sorted(os.listdir(label_dir))
    for fn in files:
        if not fn.endswith('.txt'):
            continue
        stem = os.path.splitext(fn)[0]
        ir_fname = None
        for ext in ('.jpeg', '.jpg', '.png'):
            if os.path.exists(os.path.join(ir_dir, stem + ext)):
                ir_fname = stem + ext
                break
        if ir_fname is None:
            continue
        a = extract_attributes(os.path.join(label_dir, fn), ir_fname)
        if a is not None:
            out[stem] = a
    cache_path = os.path.join(label_dir, 'attrs.json')
    with open(cache_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"{split}: {len(out)} labeled images -> {cache_path}")
    return cache_path


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', default='D:/StudyFiles/MachineLearning/datasets/FLIR-align-3class/FLIR-align-3class')
    ap.add_argument('--splits', nargs='+', default=['train', 'test'])
    args = ap.parse_args()
    for s in args.splits:
        _build_cache(s, args.data_root)
