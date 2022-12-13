import numpy as np
from scipy.spatial import KDTree

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass, base_func


class FemSolutionInterpolation():

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
        coords = self.mesh.coordinates2D[self.mesh.nodes_of_elem[elem_idx]]
        return base_func(x, coords)

    def get_element_on_position(self, x: np.ndarray):
        # TODO get k closest centers and check for inclusion in triangle
        _, elem_idx = self.kdTree.query(x)
        return elem_idx

    def set_values_to_interpolate(self, fem_solution: np.ndarray):
        self.values = fem_solution
