import time

import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import heapq

from matplotlib import tri

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass
from sources.mesh_utils import construct_nodes_adj_graph, create_nodes_to_elems_mapping


class SignedDistanceInitialization:

    def __init__(self, domain_shape: tuple, mesh: Mesh, low_density_value: float = 0.0001):
        self.dom_shape = domain_shape
        self.mesh = mesh
        self.low_density_value = low_density_value
        self.elems_centers = np.array([center_of_mass(self.mesh.coordinates2D[nodes]) for nodes in self.mesh.nodes_of_elem])
        self.node_to_elems = create_nodes_to_elems_mapping(self.mesh)
        self.nodes_adj_graph = construct_nodes_adj_graph(self.mesh)


    def __call__(self, density: np.ndarray):
        init_phi = self.compute_sign_distance_direct(density)
        init_phi_elems = np.array([
            np.average(init_phi[nodes])
            for nodes in self.mesh.nodes_of_elem
        ])
        return init_phi_elems


    def compute_sign_distance_direct(self, density: np.ndarray):
        boundary_nodes = self.get_boundary_nodes(density)
        assert len(boundary_nodes) > 0
        boundary = self.mesh.coordinates2D[boundary_nodes]
        distance = np.min(
            np.linalg.norm(
                self.mesh.coordinates2D[:, None] - boundary,
                axis=-1
            ),
            axis=-1
        )
        inner_nodes = self.get_inner_nodes(density)
        distance[inner_nodes] *= -1

        return distance

    def compute_sign_dist_step(self, density: np.ndarray):
        boundary_nodes = self.get_boundary_nodes(density)
        assert len(boundary_nodes) > 0

        distance = np.full(self.mesh.nodes_num, fill_value=-1)
        distance[boundary_nodes] = 0

        que = deque(boundary_nodes)

        while len(que) > 0:
            idx = que.popleft()
            dist = distance[idx]
            for node in self.nodes_adj_graph[idx]:
                if distance[node] != -1:
                    continue
                distance[node] = dist + 1
                que.append(node)

        inner_nodes = self.get_inner_nodes(density)
        distance[inner_nodes] *= -1
        return distance

    def compute_sign_dist(self, density):
        boundary_nodes = self.get_boundary_nodes(density)
        assert len(boundary_nodes) > 0

        distance = np.full(self.mesh.nodes_num, fill_value=np.inf)
        distance[boundary_nodes] = 0

        que = [
            (np.linalg.norm(self.mesh.coordinates2D[s] - self.mesh.coordinates2D[x]), x)
            for s in boundary_nodes for x in self.nodes_adj_graph[s]
        ]
        while len(que) > 0:
            dist, idx = heapq.heappop(que)
            if dist < distance[idx]:
                distance[idx] = dist
                for neigh in self.nodes_adj_graph[idx]:
                    extra_dist = np.linalg.norm(self.mesh.coordinates2D[idx] - self.mesh.coordinates2D[neigh])
                    heapq.heappush(que, (dist + extra_dist, neigh))

        inner_nodes = self.get_inner_nodes(density)
        distance[inner_nodes] *= -1

        return distance

    def init_domain_with_holes(self, holes_per_axis: tuple, radius: float):
        phi = -np.cos(self.elems_centers[:, 0] / self.dom_shape[0] * holes_per_axis[0] * 2 * np.pi) \
              * np.cos(self.elems_centers[:, 1] / self.dom_shape[1] * holes_per_axis[1] * 2 * np.pi) \
              + radius - 1
        return np.where(phi < 0, 1., self.low_density_value)

    def get_boundary_nodes(self, density: np.ndarray):
        return [
            v for v in range(self.mesh.nodes_num)
            if len({density[elem] for elem in self.node_to_elems[v]}) == 2
        ]

    def get_inner_nodes(self, density: np.ndarray):
        return [
            v for v in range(self.mesh.nodes_num)
            if all(density[elem] == 1 for elem in self.node_to_elems[v])
        ]

    def is_neumann_boundary(self, node_ids):
        for n_idx in node_ids:
            # if n_idx in self.mesh.dirichlet_boundaries or n_idx in self.mesh.neumann_boundaries:
            if n_idx in self.mesh.neumann_boundaries:
                return True
        return False

if __name__ == '__main__':

    mesh_path = 'sources/examples/meshes/rectangle180x60v4.msh'
    mesh = Mesh(mesh_path)
    shape = (180, 60)
    sd = SignedDistanceInitialization(domain_shape=shape, mesh=mesh)

    density = sd.init_domain_with_holes((4, 2), 0.5)

    start = time.time()
    exact = sd.compute_sign_distance_direct(density)
    exact_time = time.time() - start

    start = time.time()
    bfs = sd.compute_sign_dist_step(density)
    bfs_time = time.time() - start

    start = time.time()
    dijkstra = sd.compute_sign_dist(density)
    dijkstra_time = time.time() - start


    print(f'exact: {exact_time}')
    print(f'bfs : {bfs_time}')
    print(f'dijkstra : {dijkstra_time}')

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, exact)
    plt.title('exact')
    plt.colorbar()
    plt.show()

    plt.tripcolor(triangulation, bfs)
    plt.title('fast')
    plt.colorbar()
    plt.show()

    plt.tripcolor(triangulation, dijkstra)
    plt.title('dijkstra')
    plt.colorbar()
    plt.show()

    plt.tripcolor(triangulation, dijkstra - exact)
    plt.title('difference')
    plt.colorbar()
    plt.show()
