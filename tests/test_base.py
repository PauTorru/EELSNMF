import hyperspy.api as hs
import numpy as np

import EELSNMF as enmf


def test_eelsnmf_basic():
    # Create a small dummy 3D datacube (2x2 spatial, 100 spectral points)
    # Ensure values are strictly positive for NMF
    np.random.seed(42)
    data = np.random.rand(2, 2, 100) + 1.0

    # Create a hyperspy Signal1D
    s = hs.signals.Signal1D(data)

    # Set the energy axis properties
    s.axes_manager[-1].offset = 500.0
    s.axes_manager[-1].scale = 1.0
    s.axes_manager[-1].units = "eV"
    s.axes_manager[-1].name = "energy"

    # Initialize the EELSNMF class
    # Explicitly pass E0, alpha, beta to avoid metadata requirement
    deco = enmf.EELSNMF(s, E0=300000.0, alpha=0.02, beta=0.036)

    # Check that parameters were correctly assigned
    assert deco.E0 == 300000.0
    assert deco.alpha == 0.02
    assert deco.beta == 0.036

    # Build G matrix with fine structure range for O_K (within 500-600 eV)
    deco.build_G(low_loss=None, fine_structure_ranges={"O_K": (526.0, 560.0)})

    # Ensure G is initialized and has the correct shape
    assert hasattr(deco, "G")
    assert deco.G is not None
    assert deco.G.shape[0] == 100  # spectral dimension

    # Run the default decomposition for a few iterations
    n_components = 2
    deco.decomposition(
        n_components=n_components,
        max_iters=10,
        tol=1e-9,
        decomposition_method="default_decomposition",
    )

    # Check that W and H are created and have the correct shapes
    assert hasattr(deco, "W")
    assert hasattr(deco, "H")
    assert deco.W is not None
    assert deco.H is not None

    # X ~ G * W * H
    # X shape is (energy_size, spatial_pixels) = (100, 4)
    # G shape is (energy_size, G_cols) = (100, G_cols)
    # W shape is (G_cols, n_components)
    # H shape is (n_components, spatial_pixels) = (2, 4)
    assert deco.W.shape[1] == n_components
    assert deco.H.shape[0] == n_components
    assert deco.H.shape[1] == 4
