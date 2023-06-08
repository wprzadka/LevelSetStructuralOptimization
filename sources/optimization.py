from enum import Enum
from typing import Tuple, Callable
import numpy as np
from matplotlib import tri, pyplot as plt

from SimpleFEM.source.examples.materials import MaterialProperty
from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM
from SimpleFEM.source.utilities.computation_utils import center_of_mass, area_of_triangle
from sources.finite_difference import FiniteDifference
from sources.radial_base_functions import RadialBaseFunctions
from sources.signed_distance import SignedDistanceInitialization


class LevelSetUpdaterType(Enum):
    FINITE_DIFFERENCE = 0
    RADIAL_BASE_FUNCTIONS = 1


class LevelSetMethod:

    def __init__(
            self,
            mesh: Mesh,
            mesh_shape: Tuple[float, float],
            material: MaterialProperty,
            rhs_func: Callable,
            dirichlet_func: Callable = None,
            neumann_func: Callable = None,
            updater_type: LevelSetUpdaterType = LevelSetUpdaterType.RADIAL_BASE_FUNCTIONS
    ):
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.material = material

        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

        self.updater_type = updater_type
        self.elem_volumes = self.get_elems_volumes()
        self.low_density_value = 0.001

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
        sign_dist_init = SignedDistanceInitialization(
            domain_type='mesh',
            domain=self.mesh,
            domain_shape=self.mesh_shape,
            low_density_value=self.low_density_value
        )
        density = sign_dist_init.fill_uniformly_with_holes(holes_per_axis=(6, 3), radius=min(*self.mesh_shape) / 10)

        triangulation = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )
        plt.tripcolor(triangulation, density, cmap='gray_r')
        plt.show()

        init_phi = sign_dist_init(density)
        init_phi_elems = np.array([
            np.average(init_phi[nodes])
            for nodes in self.mesh.nodes_of_elem
        ])

        triangulation = tri.Triangulation(
            x=self.mesh.coordinates2D[:, 0],
            y=self.mesh.coordinates2D[:, 1],
            triangles=self.mesh.nodes_of_elem
        )
        plt.tricontour(triangulation, init_phi, levels=[0])
        plt.show()

        # compute local stiffness matrices per element
        elems_stiff_mat = np.array([fem.construct_local_stiffness_matrix(el_idx) for el_idx in range(self.mesh.elems_num)])
        # compute centers of elements to density computation
        elems_centers = np.array([center_of_mass(self.mesh.coordinates2D[nodes]) for nodes in self.mesh.nodes_of_elem])

        if self.updater_type == LevelSetUpdaterType.RADIAL_BASE_FUNCTIONS:
            phi = RadialBaseFunctions(self.mesh, points=elems_centers, init_values=init_phi_elems)
        elif self.updater_type == LevelSetUpdaterType.FINITE_DIFFERENCE:
            phi = FiniteDifference(self.mesh, shape=self.mesh_shape, level_set_vals=init_phi_elems, space_delta=0.5)
        else:
            raise Exception(f'Unknown method {self.updater_type}. Use one of [{", ".join(map(str,LevelSetUpdaterType))}].')

        for i in range(iteration_limit):
            # compute v s.t. J'(\Omega) = \int_{\partial\Omega} v \Theta n = 0

            # compute compliance
            displacement = fem.solve(modifier=density)
            elems_compliance = self.compliance(density, displacement, elem_stiff=elems_stiff_mat)
            # compute volume
            # elems_weights = density * self.elem_volumes

            # v_function = elems_weights - elems_compliance
            v_function = 30 - elems_compliance
            print('value function computed')

            # todo find new \phi as solution of HJB d\phi/dt - v |\nabla_x \phi| = 0
            # finite_difference = FiniteDifference(self.mesh, phi, )

            phi.update(v_function, 0.1)
            print('HJB update')

            # update density based on phi
            density = np.array([1. if phi(x) < 0 else self.low_density_value for x in elems_centers])

            if True:
                triangulation = tri.Triangulation(
                    x=self.mesh.coordinates2D[:, 0],
                    y=self.mesh.coordinates2D[:, 1],
                    triangles=self.mesh.nodes_of_elem
                )
                plt.tripcolor(triangulation, density, cmap='gray_r')
                plt.title(f"density_{i}")
                plt.show()
