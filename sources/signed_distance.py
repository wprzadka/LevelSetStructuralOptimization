import time

import matplotlib.pyplot as plt
import numpy as np
from collections import deque

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
        init_phi = self.compute_sign_dist(density)
        init_phi_elems = np.array([
            np.average(init_phi[nodes])
            for nodes in self.mesh.nodes_of_elem
        ])
        return init_phi_elems


    def compute_sign_distance_direct(self, density: np.ndarray):
        boundary_nodes = [
            v for v in range(self.mesh.nodes_num)
            if len({density[elem] for elem in self.node_to_elems[v]}) == 2
        ]
        assert len(boundary_nodes) > 0
        inner_nodes = [
            v for v in range(self.mesh.nodes_num)
            if all(density[elem] == 1 for elem in self.node_to_elems[v])
        ]
        boundary = self.mesh.coordinates2D[boundary_nodes]

        distance = np.min(
            np.linalg.norm(self.mesh.coordinates2D.reshape(-1, 1, 2) - boundary.reshape(1, -1, 2), axis=-1), axis=-1
        )
        distance[inner_nodes] *= -1

        return distance

    def compute_sign_dist(self, density: np.ndarray):
        boundary_nodes = [
            v for v in range(self.mesh.nodes_num)
            if len({density[elem] for elem in self.node_to_elems[v]}) == 2
        ]
        assert len(boundary_nodes) > 0
        inner_nodes = [
            v for v in range(self.mesh.nodes_num)
            if all(density[elem] == 1 for elem in self.node_to_elems[v])
        ]

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
    fast = sd.compute_sign_dist(density)
    fast_time = time.time() - start

    print(f'exact: {exact_time}')
    print(f'fast : {fast_time}')

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, exact)
    plt.title('exact')
    plt.colorbar()
    plt.show()
    plt.tripcolor(triangulation, fast)
    plt.title('fast')
    plt.colorbar()
    plt.show()
