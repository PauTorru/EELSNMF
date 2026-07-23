import hyperspy.api as hs
import numpy as np

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


def test_decomposition_init_options():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})

    # Test init_nmf='nndsvd'
    deco.decomposition(
        n_components=2,
        max_iters=5,
        init_nmf="nndsvd",
        decomposition_method="default_decomposition",
    )
    assert deco.W is not None
    assert deco.H is not None

    # Test random seed
    deco.decomposition(
        n_components=2,
        max_iters=5,
        init_nmf="random",
        random_state_nmf=123,
        decomposition_method="default_decomposition",
    )
    assert deco.W is not None
    assert deco.H is not None


def test_decomposition_fixed_W():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})

    n_comp = 2
    G_cols = deco.G.shape[1]

    # Create W_fixed_bool and W_fixed_values
    W_fixed_bool = np.zeros((G_cols, n_comp), dtype=bool)
    W_fixed_values = np.zeros((G_cols, n_comp), dtype=float)

    # Fix background columns to 1.0
    W_fixed_bool[: deco.n_background, :] = True
    W_fixed_values[: deco.n_background, :] = 1.0

    deco.decomposition(
        n_components=n_comp,
        max_iters=5,
        W_fixed_bool=W_fixed_bool,
        W_fixed_values=W_fixed_values,
        decomposition_method="default_decomposition",
        rescale_G=False,
    )

    assert deco.W is not None
    # Check that fixed values remained set
    np.testing.assert_allclose(deco.W[: deco.n_background, :], 1.0)


def test_decomposition_custom_initialization():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})

    n_comp = 2
    G_cols = deco.G.shape[1]
    n_spatial = deco.X.shape[1]

    W_init = np.random.rand(G_cols, n_comp) + 0.1
    H_init = np.random.rand(n_comp, n_spatial) + 0.1

    deco.decomposition(
        n_components=n_comp,
        max_iters=5,
        W_init=W_init,
        H_init=H_init,
        decomposition_method="default_decomposition",
    )

    assert deco.W.shape == (G_cols, n_comp)
    assert deco.H.shape == (n_comp, n_spatial)


def test_default_kl_decomposition():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})

    deco.decomposition(
        n_components=2,
        max_iters=5,
        decomposition_method="default_kl_decomposition",
    )

    assert deco.W is not None
    assert deco.H is not None


def test_get_model_and_enforce_dtype():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    model = deco.get_model()
    assert model.shape == (100, 4)

    deco.dtype = np.float32
    deco.enforce_dtype()
    assert deco.X.dtype == np.float32
    assert deco.W.dtype == np.float32
    assert deco.H.dtype == np.float32
