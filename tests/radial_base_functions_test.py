import numpy as np
import pytest

from SimpleFEM.source.mesh import Mesh
from sources.radial_base_functions import RadialBaseFunctions
from tests.mesh_mock import MeshFixture


@pytest.mark.parametrize("mesh_path, func", [
    (
            'sources/examples/truss.msh',
            lambda x: np.sum(np.sin(x ** 2), axis=1)
    )
])
def test_rbf_values_on_mesh_nodes(mesh_path, func):
    mesh = Mesh(mesh_path)
    exact = func(mesh.coordinates2D)

    rbf_interpolation = RadialBaseFunctions(mesh, exact, dims=2)
    values = np.array([rbf_interpolation(x) for x in mesh.coordinates2D])

    assert np.allclose(values, exact)
