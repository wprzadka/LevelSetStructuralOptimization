import numpy as np
from scipy.spatial import KDTree

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass, base_func, interp_value


class FemSolutionInterpolation:

    def __init__(self, mesh: Mesh):
        self.mesh = mesh
        self.kdTree = KDTree([center_of_mass(v) for v in self.mesh.coordinates2D[self.mesh.nodes_of_elem]])
        self.values = None

    def __call__(self, x: np.ndarray):
        if self.values is None:
            raise Exception('FEM solution is not set. Call \'set_values_to_interpolate(fem_solution: np.ndarray)\' '
                            'before requesting interpolated values')
        return self.get_value(x)

    def get_value(self, x: np.ndarray):
        elem_idx = self.get_element_on_position(x)
        nodes = self.mesh.nodes_of_elem[elem_idx]
        coords = self.mesh.coordinates2D[nodes]
        vals = self.values[nodes]
        return interp_value(x, coords, vals)

    """
    :x: coordinate for which element has to be found
    :k: number of closest element centers to check for inclusion of x
    """
    def get_element_on_position(self, x: np.ndarray, k: int = 6):
        _, elem_ids = self.kdTree.query(x, k)
        for e_idx in elem_ids:
            if self.isInsideElement(x, e_idx):
                return e_idx
        raise Exception("No element contains the point")

    def isInsideElement(self, point: np.ndarray, elem_idx: int):
        coords = self.mesh.coordinates2D[self.mesh.nodes_of_elem[elem_idx]]
        edges = np.array([coords[i] - coords[i - 1] for i in range(0, 3)])
        inner = np.array([point - coords[i - 1] for i in range(0, 3)])
        signs = np.sign(np.sum(edges * inner, axis=1))
        return 0 in signs or np.all(signs == signs[0])

    def set_values_to_interpolate(self, fem_solution: np.ndarray):
        self.values = fem_solution
