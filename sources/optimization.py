from enum import Enum
from typing import Tuple, Callable
import numpy as np
from matplotlib import tri, pyplot as plt

from SimpleFEM.source.examples.materials import MaterialProperty
from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM
from SimpleFEM.source.utilities.computation_utils import area_of_triangle
from sources.finite_difference import FiniteDifference
from sources.mesh_utils import construct_elems_adj_graph
from sources.rbf_filter_proxy import RbfFilterPoints
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
            updater_type: LevelSetUpdaterType = LevelSetUpdaterType.RADIAL_BASE_FUNCTIONS,
    ):
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.material = material

        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

        self.updater_type = updater_type
        self.elem_volumes = self.get_elems_volumes()
        self.low_density_value = 1e-4

        self.adj_elems = construct_elems_adj_graph(mesh)


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
        return elements_compliance * density


    def smoothness_filter(self, vals: np.ndarray):
        new_vals = np.empty_like(vals)
        for el_idx, adj in enumerate(self.adj_elems):
            adj_vals = vals[adj]
            adj_len = adj_vals.size
            new_vals[el_idx] = (adj_len * vals[el_idx] + np.sum(adj_vals)) / (2 * adj_len)
        return new_vals

    def optimize(self, iteration_limit: int, lag_mult: float):
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
            domain_shape=self.mesh_shape,
            mesh=self.mesh,
            low_density_value=self.low_density_value
        )
        density = sign_dist_init.init_domain_with_holes((4, 2), 0.6)
        init_phi_elems = sign_dist_init(density)

        if __debug__:
            triangulation = tri.Triangulation(
                x=self.mesh.coordinates2D[:, 0],
                y=self.mesh.coordinates2D[:, 1],
                triangles=self.mesh.nodes_of_elem
            )
            plt.tripcolor(triangulation, density, cmap='gray_r')
            plt.title("initial holes")
            plt.show()

            plt.tripcolor(triangulation, init_phi_elems)
            plt.colorbar()
            plt.title("initial level set values")
            plt.show()

        # compute local stiffness matrices per element
        elems_stiff_mat = np.array([fem.construct_local_stiffness_matrix(el_idx) for el_idx in range(self.mesh.elems_num)])

        if self.updater_type == LevelSetUpdaterType.RADIAL_BASE_FUNCTIONS:
            # phi = RadialBaseFunctions(points=elems_centers, init_values=init_phi_elems)
            phi = RbfFilterPoints(points_ratio=0.2, points=elems_centers, init_values=init_phi_elems)
        elif self.updater_type == LevelSetUpdaterType.FINITE_DIFFERENCE:
            phi = FiniteDifference(self.mesh, shape=self.mesh_shape, level_set_vals=init_phi_elems, space_delta=0.5)
        else:
            raise Exception(f'Unknown method {self.updater_type}. Use one of [{", ".join(map(str,LevelSetUpdaterType))}].')

        history = {'cost': [], 'compliance': [], "weight": []}

        for i in range(1, iteration_limit):
            print(f"iteration {i}")
            # compute v s.t. J'(\Omega) = \int_{\partial\Omega} v \Theta n = 0

            # compute compliance
            displacement = fem.solve(modifier=density)
            elems_compliance = self.compliance(density, displacement, elem_stiff=elems_stiff_mat)

            weight = np.sum(density * self.elem_volumes)
            compliance = np.sum(elems_compliance)

            history['cost'].append(compliance + lag_mult * weight)
            history['compliance'].append(compliance)
            history['weight'].append(weight)

            v_function = elems_compliance - lag_mult

            print('value function computed')

            v_function_filtered = self.smoothness_filter(v_function)

            for _ in range(10):
                phi.update(v_function_filtered, dt = 1 / np.max(np.abs(v_function_filtered)))
            print('HJB update')

            # update density based on phi
            phi_values = np.array([phi(x) for x in sign_dist_init.elems_centers])
            density = np.where(phi_values < 0, 1., self.low_density_value)

            if i > 0 and i % 5 == 0:
                init_phi_elems = sign_dist_init(density)
                phi.reinitialize(init_phi_elems)

            if __debug__ or i == iteration_limit - 1:

                print(elems_compliance.max(), ' / ', elems_compliance.min())

                triangulation = tri.Triangulation(
                    x=self.mesh.coordinates2D[:, 0],
                    y=self.mesh.coordinates2D[:, 1],
                    triangles=self.mesh.nodes_of_elem
                )

                plt.tripcolor(triangulation, v_function)
                plt.title('velocity function')
                plt.colorbar()
                plt.show()

                plt.tripcolor(triangulation, v_function_filtered)
                plt.title('velocity function filtered')
                plt.colorbar()
                plt.savefig(f'plots/v_func{i}')
                plt.show()

                triangulation.set_mask(density < 0.5)
                plt.tripcolor(triangulation, elems_compliance)
                plt.colorbar()

                vals = np.array([phi(x) for x in self.mesh.coordinates2D])
                plt.tricontour(triangulation, vals)
                plt.colorbar()
                plt.title(f"compliance_{i}")
                plt.savefig(f'plots/compl{i}')
                plt.show()

                plot_displ(self.mesh, displacement, density, scale_factor=1)

                fig, axs = plt.subplots(3, 1)
                for i, lab in enumerate(history.keys()):
                    axs[i].plot(history[lab])
                    axs[i].set_ylabel(lab)
                    axs[i].set_xlabel("iteration")
                    axs[i].grid()
                plt.savefig('plots/history')
                plt.show()
                plt.close(fig)

def plot_displ(mesh, displ, density, scale_factor = 1e2):
    half = len(displ) // 2
    displacements = scale_factor * np.vstack((displ[:half], displ[half:])).T

    before = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    before.set_mask(density < 0.5)
    plt.triplot(before, color='#1f77b4')

    after = tri.Triangulation(
        x=mesh.coordinates2D[:, 0] + displacements[:, 0],
        y=mesh.coordinates2D[:, 1] + displacements[:, 1],
        triangles=mesh.nodes_of_elem
    )
    after.set_mask(density < 0.5)
    plt.triplot(after, color='#ff7f0e')
    plt.title("displacements")
    plt.grid()
    plt.show()