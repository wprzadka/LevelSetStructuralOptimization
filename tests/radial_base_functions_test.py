import numpy as np
import pytest

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass
from sources.radial_base_functions import RadialBaseFunctions
from tests.mesh_mock import MeshFixture


@pytest.mark.parametrize("mesh_path, func", [
    (
            'sources/examples/meshes/rectangle180x60v4.msh',
            lambda x: np.sum(np.sin(x ** 2), axis=1)
    )
])
def test_rbf_values_on_element_centers(mesh_path, func):
    mesh = Mesh(mesh_path)
    points = np.array([
        center_of_mass(mesh.coordinates2D[nodes])
        for nodes in mesh.nodes_of_elem
    ])
    exact = func(points)

    rbf_interpolation = RadialBaseFunctions(points, exact, dims=2)
    values = np.array([rbf_interpolation(x) for x in points])

    assert np.allclose(values, exact)
