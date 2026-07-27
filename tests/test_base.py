import os
import tempfile

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


def test_eelsnmf_basic():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)

    assert deco.E0 == 300000.0
    assert deco.alpha == 0.02
    assert deco.beta == 0.036

    deco.build_G(low_loss=None, fine_structure_ranges={"O_K": (526.0, 560.0)})

    assert hasattr(deco, "G")
    assert deco.G is not None
    assert deco.G.shape[0] == 100

    n_components = 2
    deco.decomposition(
        n_components=n_components,
        max_iters=10,
        tol=1e-9,
        decomposition_method="default_decomposition",
    )

    assert hasattr(deco, "W")
    assert hasattr(deco, "H")
    assert deco.W is not None
    assert deco.H is not None

    assert deco.W.shape[1] == n_components
    assert deco.H.shape[0] == n_components
    assert deco.H.shape[1] == 4


def test_eelsnmf_metadata_init():
    s = create_dummy_signal()
    # Add metadata for TEM acquisition parameters
    s.metadata.add_node("Acquisition_instrument.TEM")
    s.metadata.Acquisition_instrument.TEM.beam_energy = 300.0  # in kV
    s.metadata.Acquisition_instrument.TEM.convergence_angle = 20.0  # in mrad
    s.metadata.add_node("Acquisition_instrument.TEM.Detector.EELS")
    s.metadata.Acquisition_instrument.TEM.Detector.EELS.collection_angle = (
        36.0  # in mrad
    )

    deco = enmf.EELSNMF(s, E0=None, alpha=None, beta=None)
    assert np.isclose(deco.E0, 300000.0)
    assert np.isclose(deco.alpha, 0.02)
    assert np.isclose(deco.beta, 0.036)


def test_change_dtype():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(low_loss=None, fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    deco.change_dtype(np.float32)
    assert deco.X.dtype == np.float32
    assert deco.G.dtype == np.float32
    assert deco.W.dtype == np.float32
    assert deco.H.dtype == np.float32


def test_temp_arrays():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)

    dummy_data = np.ones((5, 5))
    deco.create_temp_array("test_arr", dummy_data)
    assert hasattr(deco, "test_arr")
    assert "test_arr" in deco._temp_arrays
    assert "test_arr" in deco._m

    deco.delete_temp_arrays()
    assert not hasattr(deco, "test_arr")
    assert "test_arr" not in deco._temp_arrays
    assert "test_arr" not in deco._m


def test_save_and_load():
    s = create_dummy_signal()
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)
    deco.build_G(low_loss=None, fine_structure_ranges={"O_K": (526.0, 560.0)})
    deco.decomposition(
        n_components=2, max_iters=5, decomposition_method="default_decomposition"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        file_base = "my_eelsnmf_model"
        deco.save(file_base, path=tmpdir, overwrite=True)

        full_pkl_path = os.path.join(tmpdir, file_base + ".pkl")
        assert os.path.exists(full_pkl_path)

        loaded_deco = enmf.load(full_pkl_path)
        assert loaded_deco.n_components == 2
        assert loaded_deco.W.shape == deco.W.shape
        assert loaded_deco.H.shape == deco.H.shape
