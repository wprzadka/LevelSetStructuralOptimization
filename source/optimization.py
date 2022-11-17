from typing import Tuple
import numpy as np

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass


class LevelSetMethod():

    def __init__(self, mesh: Mesh, mesh_shape: Tuple[float, float]):
        self.mesh = mesh
        self.mesh_shape = mesh_shape

    def fill_uniformly_with_holes(self, holes_per_axis: tuple, radius: float) -> np.ndarray:

        border_dist = tuple(dim_size / (holes + 2) for dim_size, holes in zip(self.mesh_shape, holes_per_axis))
        x_points = np.linspace(border_dist[0], self.mesh_shape[0] - border_dist[0], holes_per_axis[0])
        y_points = np.linspace(border_dist[1], self.mesh_shape[1] - border_dist[1], holes_per_axis[1])

        centers = np.array([[x, y] for x in x_points for y in y_points])

        density = np.ones(self.mesh.elems_num)
        radius_sqr = radius ** 2
        for elem_idx, nodes in enumerate(self.mesh.nodes_of_elem):
            point = center_of_mass(self.mesh.coordinates2D[nodes])
            dist_sqr = np.sum((centers - point) ** 2, 1)
            if np.any(dist_sqr < radius_sqr):
                density[elem_idx] = 0

        return density