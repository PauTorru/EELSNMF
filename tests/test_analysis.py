import hyperspy.api as hs
import numpy as np
import pandas as pd

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


def test_calculate_loadings():
    s = create_dummy_signal(shape=(2, 3, 100))
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    deco.calculate_loadings()
    assert hasattr(deco, "loadings")
    assert deco.loadings.shape == (2, 2, 3)


def test_get_edge_from_component():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    edge_sig = deco.get_edge_from_component(component_id=0, edge="O_K")
    assert isinstance(edge_sig, hs.signals.Signal1D)
    assert edge_sig.data.shape == (100,)


def test_quantify_components():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    q_arr = deco.quantify_components()
    assert q_arr.shape == (1, 2)
    assert hasattr(deco, "component_quantification")
    assert isinstance(deco.component_quantification, pd.DataFrame)
    assert deco.component_quantification.shape == (1, 2)


def test_spatial_standard_q_and_get_chemical_maps():
    s = create_dummy_signal(shape=(2, 2, 100))
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    sq = deco.spatial_standard_q()
    assert sq.shape == (1, 2, 2)

    cmaps = deco.get_chemical_maps(quantified=True)
    assert "O_K" in cmaps
    assert cmaps["O_K"].shape == (2, 2)


def test_evaluate_comp_contributions():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    loo, power, ev = deco.evaluate_comp_contributions()
    assert len(loo) == 2
    assert len(power) == 2
    assert len(ev) == 2

    # Test edge-specific evaluation
    loo_edge, power_edge, ev_edge = deco.evaluate_comp_contributions(edge="O_K")
    assert len(loo_edge) == 2
