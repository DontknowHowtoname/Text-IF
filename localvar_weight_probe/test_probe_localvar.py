import numpy as np
import pytest

from probe_localvar import local_variance, temp_weight


def test_local_variance_flat_image_is_zero():
    flat = np.full((32, 32), 0.5, dtype=np.float64)
    v = local_variance(flat, win=7, normalize=False)
    assert v.shape == flat.shape
    assert np.allclose(v, 0.0, atol=1e-10)


def test_local_variance_matches_manual_patch():
    rng = np.random.default_rng(0)
    img = rng.random((32, 32))
    win = 7
    pad = win // 2
    v = local_variance(img, win=win, normalize=False)
    patch = img[16 - pad:16 + pad + 1, 16 - pad:16 + pad + 1]
    assert v[16, 16] == pytest.approx(patch.var(), abs=1e-9)


def test_temp_weight_tau_large_is_half():
    rng = np.random.default_rng(1)
    a, b = rng.random((8, 8)), rng.random((8, 8))
    w = temp_weight(a, b, tau=1e6)
    assert np.allclose(w, 0.5, atol=1e-4)


def test_temp_weight_tau_small_is_argmax():
    rng = np.random.default_rng(2)
    a, b = rng.random((8, 8)), rng.random((8, 8))
    w = temp_weight(a, b, tau=1e-6)
    expected = (a > b).astype(np.float64)
    assert np.allclose(w, expected, atol=1e-4)


def test_temp_weight_is_symmetric_complement():
    rng = np.random.default_rng(3)
    a, b = rng.random((8, 8)) * 0.1, rng.random((8, 8))
    w = temp_weight(a, b, tau=0.5)
    assert np.all((w >= 0) & (w <= 1))
