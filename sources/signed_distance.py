from typing import Union
import numpy as np
from queue import PriorityQueue

from SimpleFEM.source.mesh import Mesh
from sources.mesh_utils import construct_graph, create_nodes_to_elems_mapping


class SignedDistanceInitialization:

    available_domains = ['mesh', 'grid']

    def __init__(self, domain_type: str, domain: Union[Mesh, np.ndarray]):
        if domain_type not in self.available_domains:
            raise Exception(f'domain type {domain_type} is not available. Chose one from {self.available_domains}')
        self.domain_type = domain_type

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
