from typing import Tuple, Callable
import numpy as np

from SimpleFEM.source.examples.materials import MaterialProperty
from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.fem.elasticity_setup import ElasticitySetup as FEM
from sources.history_tracker import HistoryTracker
from sources.mesh_utils import construct_elems_adj_graph
from sources.plotting_utils import PlottingUtils
from sources.rbf_filter_proxy import RbfFilterPoints
from sources.signed_distance import SignedDistanceInitialization


class LevelSetMethod:

    def __init__(
            self,
            mesh: Mesh,
            mesh_shape: Tuple[float, float],
            material: MaterialProperty,
            rhs_func: Callable,
            dirichlet_func: Callable = None,
            neumann_func: Callable = None,
            lag_mult: float = 1.,
            reinitialization_period: int = 5,
            updates_num: int = 10,
            holes_per_axis: tuple = (4, 2),
            holes_radius: float = 0.6
    ):
        self.mesh = mesh
        self.mesh_shape = mesh_shape
        self.material = material

        self.rhs_func = rhs_func
        self.dirichlet_func = dirichlet_func
        self.neumann_func = neumann_func

        self.low_density_value = 1e-4
        self.lag_mult = lag_mult
        self.reinitialization_period = reinitialization_period
        self.updates_num = updates_num
        self.holes_per_axis = holes_per_axis
        self.holes_radius = holes_radius

        self.adj_elems = construct_elems_adj_graph(mesh)

        self.plots_utils = PlottingUtils(
            mesh=self.mesh,
            shape=self.mesh_shape
        )
        self.history = HistoryTracker(
            mesh=mesh,
            lag_mult=lag_mult
        )

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
            domain_shape=self.mesh_shape,
            mesh=self.mesh,
            low_density_value=self.low_density_value
        )
        density = sign_dist_init.init_domain_with_holes(self.holes_per_axis, self.holes_radius)
        init_phi_elems = sign_dist_init(density)

        if __debug__:
            self.plots_utils.initial_domain_plot(density, init_phi_elems)

        # compute local stiffness matrices per element
        elems_stiff_mat = np.array([fem.construct_local_stiffness_matrix(el_idx) for el_idx in range(self.mesh.elems_num)])

        phi = RbfFilterPoints(points_ratio=0.3, points=sign_dist_init.elems_centers, init_values=init_phi_elems)

        for iteration in range(1, iteration_limit):
            print(f"iteration {iteration}")
            # compute compliance
            displacement = fem.solve(modifier=density)
            elems_compliance = self.compliance(density, displacement, elem_stiff=elems_stiff_mat)

            self.history.log(density, elems_compliance)
            # compute v s.t. J'(\Omega) = \int_{\partial\Omega} v \Theta n = 0
            v_function = elems_compliance - self.lag_mult
            v_function = self.smoothness_filter(v_function)
            # update phi
            for _ in range(self.updates_num):
                phi.update(v_function, dt = 1 / np.max(np.abs(v_function)))
            # update density based on phi
            phi_values = np.array([phi(x) for x in sign_dist_init.elems_centers])
            density = np.where(phi_values < 0, 1., self.low_density_value)
            # reinitialization
            if iteration > 0 and iteration % self.reinitialization_period == 0:
                init_phi_elems = sign_dist_init(density)
                phi.reinitialize(init_phi_elems)

            if iteration < 25 or iteration % 5 == 0 or iteration in [32, 64]:
                self.plots_utils.make_plots(
                    displacement,
                    density,
                    v_function,
                    iteration
                )
                self.plots_utils.plot_implicit_function(rbf=phi, file_name=f'phi/phi{iteration}')

                if iteration == 64:
                    self.plots_utils.plot_history(history=self.history.history, file_name=f'history{iteration}')
