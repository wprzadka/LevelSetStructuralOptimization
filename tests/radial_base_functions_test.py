import numpy as np
import pytest

from sources.radial_base_functions import RadialBaseFunctions
from tests.mesh_mock import MeshFixture


@pytest.mark.parametrize("coordinates2D, nodes_of_elem, func", [
    (
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
            np.array([[0, 1, 2], [1, 2, 3]]),
            lambda x: np.sum(np.sin(x ** 2), axis=1)
    )
])
def test_rbf_values_on_mesh_nodes(coordinates2D, nodes_of_elem, func):
    mesh = MeshFixture(coordinates2D, nodes_of_elem)
    exact = func(coordinates2D)

    rbf_interpolation = RadialBaseFunctions(mesh, exact, dims=2)
    values = np.array([rbf_interpolation(x) for x in coordinates2D])

    assert np.allclose(values, exact)
