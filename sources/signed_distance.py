import numpy as np
from queue import PriorityQueue

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass
from sources.mesh_utils import construct_graph, create_nodes_to_elems_mapping


class SignedDistanceInitialization:

    def __init__(self, domain_shape: tuple, mesh: Mesh, low_density_value: float = 0.0001):
        self.dom_shape = domain_shape
        self.mesh = mesh
        self.low_density_value = low_density_value
        self.elems_centers = np.array([center_of_mass(self.mesh.coordinates2D[nodes]) for nodes in self.mesh.nodes_of_elem])


    def __call__(self, density: np.ndarray):
        init_phi = self.compute_sign_distance_on_mesh(density)
        init_phi_elems = np.array([
            np.average(init_phi[nodes])
            for nodes in self.mesh.nodes_of_elem
        ])
        return init_phi_elems


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

    def init_domain_with_holes(self, holes_per_axis: tuple, radius: float):
        phi = -np.cos(self.elems_centers[:, 0] / self.dom_shape[0] * holes_per_axis[0] * 2 * np.pi) \
              * np.cos(self.elems_centers[:, 1] / self.dom_shape[1] * holes_per_axis[1] * 2 * np.pi) \
              + radius - 1
        return np.where(phi < 0, 1., self.low_density_value)

    def is_boundary(self, node_ids):
        for n_idx in node_ids:
            # if n_idx in self.mesh.dirichlet_boundaries or n_idx in self.mesh.neumann_boundaries:
            if n_idx in self.mesh.neumann_boundaries:
                return True
        return False
