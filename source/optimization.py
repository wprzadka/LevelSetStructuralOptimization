from typing import Tuple, Callable
import numpy as np
from queue import PriorityQueue


from SimpleFEM.source.examples.materials import MaterialProperty
from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM
from SimpleFEM.source.utilities.computation_utils import center_of_mass, area_of_triangle
from source.mesh_utils import construct_graph, create_nodes_to_elems_mapping


class LevelSetMethod:

    def __init__(
            self,
            mesh: Mesh,
            mesh_shape: Tuple[float, float],
            material: MaterialProperty,
            rhs_func: Callable,
            dirichlet_func: Callable = None,
            neumann_func: Callable = None,
    ):
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.material = material

        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

        self.elem_volumes = self.get_elems_volumes()

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

    def compute_sign_distance(self, density: np.ndarray):
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

    def get_elems_volumes(self):
        volumes = np.array([
            area_of_triangle(self.mesh.coordinates2D[nodes_ids])
            for nodes_ids in self.mesh.nodes_of_elem
        ])
        return volumes

    def compliance(self, density: np.ndarray, displacement: np.ndarray, elem_stiff: np.ndarray):
        elements_compliance = np.zeros_like(density)
        for elem_idx, nodes_ids in enumerate(self.mesh.nodes_of_elem):
            base_func_ids = np.hstack((nodes_ids, nodes_ids + self.mesh.nodes_num))
            elem_displacement = np.expand_dims(displacement[base_func_ids], 1)

            elements_compliance[elem_idx] = elem_displacement.T @ elem_stiff[elem_idx] @ elem_displacement
        return elements_compliance

    def optimize(self, iteration_limit: int):

        fem = FEM(
            mesh=self.mesh,
            rhs_func=self.rhs_func,
            dirichlet_func=self.dirichlet_func,
            neumann_func=self.neumann_func,
            young_modulus=self.material.value[0],
            poisson_ratio=self.material.value[1]
        )

        # initialize level sets
        density = self.fill_uniformly_with_holes(holes_per_axis=(6, 3), radius=5)
        phi = self.compute_sign_distance(density)

        # compute local stiffness matrices per element
        elems_stiff_mat = np.array([fem.construct_local_stiffness_matrix(el_idx) for el_idx in range(self.mesh.elems_num)])

        for i in range(iteration_limit):
            # compute v s.t. J'(\Omega) = \int_{\partial\Omega} v \Theta n = 0
            # compute compliance
            displacement = fem.solve(modifier=density)
            elems_compliance = self.compliance(density, displacement, elem_stiff=elems_stiff_mat)
            # compute volume
            elems_weights = density * self.elem_volumes

            v_function = elems_weights - elems_compliance
            # todo find solution of HJB d\phi/dt - v |\nabla_x \phi| = 0
