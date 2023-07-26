from typing import Union, Tuple
import numpy as np
from queue import PriorityQueue

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass
from sources.mesh_utils import construct_graph, create_nodes_to_elems_mapping


class SignedDistanceInitialization:

    available_domains = ['mesh', 'grid']

    def __init__(self, domain_type: str, domain: Union[Mesh, np.ndarray], domain_shape: tuple, low_density_value: float = 0.0001):
        if domain_type not in self.available_domains:
            raise Exception(f'domain type {domain_type} is not available. Chose one from {self.available_domains}')
        self.domain_type = domain_type
        self.domain_shape = domain_shape

        if domain_type == 'mesh':
            self.mesh = domain

        self.low_density_value = low_density_value

    def __call__(self, density: np.ndarray):
        if self.domain_type == 'mesh':
            init_phi = self.compute_sign_distance_on_mesh(density)
            init_phi_elems = np.array([
                np.average(init_phi[nodes])
                for nodes in self.mesh.nodes_of_elem
            ])
            return init_phi_elems
        elif self.domain_type == 'grid':
            raise NotImplementedError()

    def compute_sign_distance_on_mesh(self, density: np.ndarray):
        neighbours = construct_graph(self.mesh)
        node_to_elems = create_nodes_to_elems_mapping(self.mesh)
        boundary_nodes = [
            v for v in range(self.mesh.nodes_num)
            if len({density[elem] for elem in node_to_elems[v]}) == 2
        ]
        assert len(boundary_nodes) > 0
        inner_nodes = [
            v for v in range(self.mesh.nodes_num)
            if all(density[elem] == 1 for elem in node_to_elems[v])
        ]

        boundary = self.mesh.coordinates2D[boundary_nodes]

        distance = np.array([np.min(np.linalg.norm(coord - boundary, axis=1))
                             for idx, coord in enumerate(self.mesh.coordinates2D)
                             ])
        distance[inner_nodes] *= -1

        return distance

    def fill_uniformly_with_holes(self, holes_per_axis: Tuple[int, int], radius: float) -> np.ndarray:
        centers = self.get_uniform_points(holes_per_axis)
        centers = self.remove_holes_from_boundary_conditions(centers, radius)

        density = np.ones(self.mesh.elems_num)
        radius_sqr = radius ** 2
        for elem_idx, nodes in enumerate(self.mesh.nodes_of_elem):
            point = center_of_mass(self.mesh.coordinates2D[nodes])
            dist_sqr = np.sum((centers - point) ** 2, 1)
            if np.any(dist_sqr < radius_sqr):
                density[elem_idx] = self.low_density_value
        return density

    def get_uniform_points(self, holes_per_axis: Tuple[int, int], include_borders=True):
        low_limit = np.zeros(2)
        high_limit = self.domain_shape
        if not include_borders:
            border_dist = np.array([dim_size / (holes + 2) for dim_size, holes in zip(self.domain_shape, holes_per_axis)])
            low_limit += border_dist
            high_limit -= border_dist

        x_points = np.linspace(low_limit[0], high_limit[0], holes_per_axis[0])
        y_points = np.linspace(low_limit[1], high_limit[1], holes_per_axis[1])
        points = np.array([[x, y] for x in x_points for y in y_points])
        return points

    def remove_holes_from_boundary_conditions(self, holes: np.ndarray, radius: float) -> np.ndarray:
        boundary_elems_centers = [
            center_of_mass(self.mesh.coordinates2D[nodes])
            for nodes in self.mesh.nodes_of_elem if self.is_boundary(nodes)
        ]

        valid_holes = np.full(holes.shape[0], fill_value=True, dtype=bool)
        radius_sqr = radius ** 2
        for idx, center in enumerate(holes):
            dist_sqr = np.sum((center - boundary_elems_centers) ** 2, 1)
            if np.any(dist_sqr < radius_sqr):
                valid_holes[idx] = False
        return holes[valid_holes]

    def is_boundary(self, node_ids):
        for n_idx in node_ids:
            # if n_idx in self.mesh.dirichlet_boundaries or n_idx in self.mesh.neumann_boundaries:
            if n_idx in self.mesh.neumann_boundaries:
                return True
        return False
