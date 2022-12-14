import numpy as np
import pytest

from sources.fem_solution_interpolation import FemSolutionInterpolation
from tests.mesh_mock import MeshFixture


@pytest.mark.parametrize("coordinates2D, nodes_of_elem, xs, exacts", [
    (
            np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),
            np.array([[0, 1, 2], [1, 2, 3]]),
            np.array([[0.1, 0.3], [0.5, 0.2], [0.9, 0.9]]),
            np.array([0, 0, 1])
    )
])
def test_fem_solution_interpolation_points(coordinates2D, nodes_of_elem, xs, exacts):
    mesh = MeshFixture(coordinates2D, nodes_of_elem)
    interpolation = FemSolutionInterpolation(mesh)

    elem_idxs = interpolation.get_element_on_position(xs)

    assert np.all(elem_idxs == exacts)