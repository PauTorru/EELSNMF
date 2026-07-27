import hyperspy.api as hs
import numpy as np
import pytest

import EELSNMF as enmf


def create_dummy_signal(shape=(2, 2, 100), offset=500.0, scale=1.0, seed=42):
    np.random.seed(seed)
    data = np.random.rand(*shape) + 1.0
    s = hs.signals.Signal1D(data)
    s.axes_manager[-1].offset = offset
    s.axes_manager[-1].scale = scale
    s.axes_manager[-1].units = "eV"
    s.axes_manager[-1].name = "energy"
    return s


def test_build_G_deltas():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)

    # Test build_G with deltas model and Kohl cross-section
    deco.build_G(
        low_loss=None,
        fine_structure_ranges={"O_K": (526.0, 560.0)},
        backgrounds=np.linspace(1, 5, 10),
        model_type="deltas",
        xsection_type="Kohl",
    )

    assert hasattr(deco, "G")
    assert deco.G is not None
    assert deco.G.ndim == 2
    assert deco.G.shape[0] == 100
    assert "O_K" in deco.edges
    assert "O_K" in deco.model._edge_slices


def test_build_G_zezhong():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)

    # Test build_G with Zezhong cross-section (skip if pyEELSMODEL datafile is missing/corrupted)
    try:
        deco.build_G(
            low_loss=None,
            fine_structure_ranges={"O_K": (526.0, 560.0)},
            model_type="deltas",
            xsection_type="Zezhong",
        )
        assert deco.G is not None
    except OSError as e:
        pytest.skip(f"Zezhong data file unavailable or corrupted in pyEELSMODEL: {e}")


def test_build_G_background_formats():
    s = create_dummy_signal()

    # 1. Integer background count
    deco1 = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco1.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)}, backgrounds=5)
    assert deco1.n_background == 5

    # 2. None background
    deco2 = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco2.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)}, backgrounds=None)
    assert deco2.n_background == 1

    # 3. Invalid background (non-iterable, non-int, non-None) raises Exception
    deco3 = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    with pytest.raises(Exception, match="Background argument invalid"):
        deco3.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)}, backgrounds=12.34)


def test_available_models():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    models = deco.available_models
    assert "deltas" in models
    assert "convolved_single" in models
