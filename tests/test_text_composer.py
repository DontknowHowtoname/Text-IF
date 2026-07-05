"""Tests for text composition from attributes."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
from scripts.build_text_descriptions import compose_sentence, maybe_fallback, FALLBACK_TEMPLATES


def _sample_attrs(**overrides):
    base = {
        'main_class': 0, 'main_class_name': 'people',
        'distance': 'mid', 'count': 'single',
        'time_of_day': 'day', 'co_occurrence': 'alone',
        'position': 'center', 'median_area': 0.002, 'n_main': 1,
    }
    base.update(overrides)
    return base


def test_compose_returns_nonempty_string():
    attrs = _sample_attrs()
    text = compose_sentence(attrs, rng=random.Random(0))
    assert isinstance(text, str)
    assert len(text) > 20


def test_compose_contains_main_class_word():
    attrs = _sample_attrs(main_class_name='car')
    text = compose_sentence(attrs, rng=random.Random(0))
    # at least one of: car, vehicle, automobile
    assert any(w in text.lower() for w in ['car', 'vehicle', 'automobile'])


def test_diversity_20_samples_differ():
    """Same attrs, different seeds should produce varied sentences."""
    attrs = _sample_attrs()
    sentences = set()
    for seed in range(20):
        text = compose_sentence(attrs, rng=random.Random(seed))
        sentences.add(text)
    assert len(sentences) >= 5, f"Only {len(sentences)} unique out of 20"


def test_unknown_time_picks_day_or_night():
    attrs = _sample_attrs(time_of_day='unknown')
    seen = set()
    for seed in range(20):
        text = compose_sentence(attrs, rng=random.Random(seed)).lower()
        if 'daylight' in text or 'daytime' in text:
            seen.add('day')
        if 'night' in text or 'low light' in text:
            seen.add('night')
    assert 'day' in seen and 'night' in seen


def test_fallback_returns_template_or_none():
    attrs = _sample_attrs(main_class=0)
    rng_low = random.Random(0)  # might trigger or not
    out = maybe_fallback(attrs, prob=1.0, rng=random.Random(0))  # always fallback
    assert out in FALLBACK_TEMPLATES[0]
    out = maybe_fallback(attrs, prob=0.0, rng=random.Random(0))  # never fallback
    assert out is None


if __name__ == '__main__':
    test_compose_returns_nonempty_string()
    print("PASS: compose returns non-empty string")
    test_compose_contains_main_class_word()
    print("PASS: compose contains class word")
    test_diversity_20_samples_differ()
    print("PASS: diversity across seeds")
    test_unknown_time_picks_day_or_night()
    print("PASS: unknown time picks day or night")
    test_fallback_returns_template_or_none()
    print("PASS: fallback behavior")
    print("All composer tests passed.")
