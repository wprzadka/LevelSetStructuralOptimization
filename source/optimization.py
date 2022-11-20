from typing import Tuple
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib import tri
from queue import PriorityQueue

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass


class LevelSetMethod:

    def __init__(self, mesh: Mesh, mesh_shape: Tuple[float, float]):
        self.mesh = mesh
        self.mesh_shape = mesh_shape

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
        border_dist = tuple(dim_size / (holes + 2) for dim_size, holes in zip(self.mesh_shape, holes_per_axis))
        x_points = np.linspace(border_dist[0], self.mesh_shape[0] - border_dist[0], holes_per_axis[0])
        y_points = np.linspace(border_dist[1], self.mesh_shape[1] - border_dist[1], holes_per_axis[1])

        points = np.array([[x, y] for x in x_points for y in y_points])
        return points

    def construct_graph(self):
        neighbours = [set() for _ in range(self.mesh.nodes_num)]
        for nodes in self.mesh.nodes_of_elem:
            for beg, end in combinations(nodes, 2):
                neighbours[beg].add(end)
                neighbours[end].add(beg)
        neighbours = [list(v) for v in neighbours]
        return neighbours

    def create_nodes_to_elems_mapping(self):
        node_to_elems = [[] for _ in range(self.mesh.nodes_num)]
        for elem_idx, nodes in enumerate(self.mesh.nodes_of_elem):
            for n in nodes:
                node_to_elems[n].append(elem_idx)
        return node_to_elems

    def compute_sign_distance(self, density: np.ndarray):
        neighbours = self.construct_graph()
        node_to_elems = self.create_nodes_to_elems_mapping()

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
