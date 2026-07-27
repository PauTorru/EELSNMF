import os
import tempfile

import hyperspy.api as hs
import numpy as np

from EELSNMF.utils import (
    ListOfSI,
    all_arrays_equal,
    convergent_factor,
    convergent_psi,
    convolve,
    find_2factors,
    find_index,
    integral_over_th,
    load_ListOfSI,
    match_axis,
    moving_average,
    norm,
    psi,
    theta_E,
)


def test_norm():
    arr = np.array([2.0, 4.0, 6.0, 10.0])
    res = norm(arr)
    assert np.isclose(res.min(), 0.0)
    assert np.isclose(res.max(), 1.0)
    assert np.allclose(res, np.array([0.0, 0.25, 0.5, 1.0]))


def test_find_2factors():
    assert find_2factors(4) == (2, 2)
    r, c = find_2factors(6)
    assert r * c == 6
    r, c = find_2factors(12)
    assert r * c == 12


def test_find_index():
    ax = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
    # Scalar lookup
    idx = find_index(ax, 205.0)
    assert idx == 1

    # Iterable lookup
    idxs = find_index(ax, [102.0, 490.0])
    assert idxs == [0, 4]


def test_convolve():
    # Length must be even
    a = np.zeros(10)
    a[5] = 1.0
    b = np.zeros(10)
    b[5] = 1.0

    conv_res = convolve(a, b)
    assert conv_res.shape == a.shape
    assert np.sum(conv_res) > 0


def test_moving_average():
    a = np.ones((2, 2, 7))
    ma = moving_average(a, n=3)
    assert ma.shape == a.shape


def test_all_arrays_equal():
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    arr3 = np.array([1, 2, 4])

    assert all_arrays_equal([arr1, arr2]) is True
    assert all_arrays_equal([arr1, arr3]) is False


def test_match_axis():
    data = np.linspace(10, 50, 100)
    s = hs.signals.Signal1D(data)
    s.axes_manager[-1].offset = 100.0
    s.axes_manager[-1].scale = 1.0

    new_axis = np.linspace(110.0, 180.0, 71)
    matched_s = match_axis(s, new_axis)
    assert isinstance(matched_s, hs.signals.Signal1D)
    assert matched_s.data.shape == (71,)
    assert np.isclose(matched_s.axes_manager[-1].offset, 110.0)


def test_physics_utils():
    # Test theta_E
    th = theta_E(500.0, 300e3)
    assert th > 0

    # Test convergent factor
    th_arr = np.linspace(0, 0.02, 100)
    f = convergent_factor(th_arr, 0.01, 0.02)
    assert len(f) == 100
    assert not np.isnan(f).any()

    # Test integral_over_th and convergent_psi
    thE_arr = theta_E(np.array([500.0, 600.0]), 300e3)
    integ = integral_over_th(thE_arr, 0.01, 0.02, n_points=100)
    assert len(integ) == 2

    cpsi = convergent_psi(np.array([500.0, 600.0]), 0.01, 0.02, kV=300e3, n_points=100)
    assert len(cpsi) == 2

    # Test psi
    p_val = psi(np.array([500.0, 600.0]), 0.02, kV=300e3)
    assert len(p_val) == 2


def test_list_of_si_and_save_load():
    np.random.seed(42)
    s1 = hs.signals.Signal1D(np.random.rand(2, 2, 50) + 1.0)
    s1.axes_manager[-1].offset = 500.0
    s1.axes_manager[-1].scale = 1.0

    s2 = hs.signals.Signal1D(np.random.rand(3, 2, 50) + 1.0)
    s2.axes_manager[-1].offset = 500.0
    s2.axes_manager[-1].scale = 1.0

    lsi = ListOfSI([s1, s2])
    assert lsi.len == 2
    assert lsi.unfolded_data.shape == (4 + 6, 50)

    # Test fold_array
    folded = lsi.fold_array(lsi.unfolded_data)
    assert len(folded) == 2
    assert folded[0].shape == (2, 2, 50)
    assert folded[1].shape == (3, 2, 50)

    # Test save and load_ListOfSI
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_lsi")
        lsi.save(save_path, overwrite=True)

        loaded_lsi = load_ListOfSI(save_path)
        assert loaded_lsi.unfolded_data.shape == lsi.unfolded_data.shape
