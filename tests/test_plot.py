import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import hyperspy.api as hs
import matplotlib.pyplot as plt

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


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


def test_plots():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    # Test plot_factors
    deco.plot_factors()

    # Test plot_loadings
    deco.plot_loadings()

    # Test plot_edges
    deco.plot_edges(normalize=False)
    deco.plot_edges(normalize=True)

    # Test plot_chemical_maps
    deco.plot_chemical_maps()

    # Test plot_average_model
    deco.plot_average_model()

    # Test plot_energy_ranges
    deco.plot_energy_ranges()
