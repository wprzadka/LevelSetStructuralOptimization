from typing import Union, Tuple
import numpy as np
from queue import PriorityQueue

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass
from sources.mesh_utils import construct_graph, create_nodes_to_elems_mapping


class SignedDistanceInitialization:

    available_domains = ['mesh', 'grid']

    def __init__(self, domain_type: str, domain: Union[Mesh, np.ndarray], domain_shape: tuple):
        if domain_type not in self.available_domains:
            raise Exception(f'domain type {domain_type} is not available. Chose one from {self.available_domains}')
        self.domain_type = domain_type
        self.domain_shape = domain_shape

        if domain_type == 'mesh':
            self.mesh = domain

    def __call__(self, density: np.ndarray):
        if self.domain_type == 'mesh':
            return self.compute_sign_distance_on_mesh(density)
        elif self.domain_type == 'grid':
            raise NotImplementedError()

    def compute_sign_distance_on_mesh(self, density: np.ndarray):
        neighbours = construct_graph(self.mesh)
        node_to_elems = create_nodes_to_elems_mapping(self.mesh)
        boundary_nodes = [
            v for v in range(self.mesh.nodes_num)
            if len({density[elem] for elem in node_to_elems[v]}) == 2
        ]
        inner_nodes = [
            v for v in range(self.mesh.nodes_num)
            if {density[elem] for elem in node_to_elems[v]} == {0}
        ]
        que = PriorityQueue()
        for node in boundary_nodes:
            que.put((0, node))

        distance = np.empty(shape=self.mesh.nodes_num)

        visited = np.full_like(distance, fill_value=False, dtype=bool)
        visited[boundary_nodes] = True

        while not que.empty():
            dist, node = que.get()
            distance[node] = dist

            for x in neighbours[node]:
                if visited[x]:
                    continue
                local_dist = np.linalg.norm(self.mesh.coordinates2D[x] - self.mesh.coordinates2D[node])
                que.put((dist + local_dist, x))
                visited[x] = True
        distance[inner_nodes] *= -1

        return distance

    def fill_uniformly_with_holes(self, holes_per_axis: Tuple[int, int], radius: float) -> np.ndarray:
        centers = self.get_uniform_points(holes_per_axis)
        density = np.ones(self.mesh.elems_num)
        radius_sqr = radius ** 2
        for elem_idx, nodes in enumerate(self.mesh.nodes_of_elem):
            point = center_of_mass(self.mesh.coordinates2D[nodes])
            dist_sqr = np.sum((centers - point) ** 2, 1)
            if np.any(dist_sqr < radius_sqr):
                density[elem_idx] = 0
        return density

    def get_uniform_points(self, holes_per_axis: Tuple[int, int]):
        border_dist = tuple(dim_size / (holes + 2) for dim_size, holes in zip(self.domain_shape, holes_per_axis))
        x_points = np.linspace(border_dist[0], self.domain_shape[0] - border_dist[0], holes_per_axis[0])
        y_points = np.linspace(border_dist[1], self.domain_shape[1] - border_dist[1], holes_per_axis[1])

        points = np.array([[x, y] for x in x_points for y in y_points])
        return points
