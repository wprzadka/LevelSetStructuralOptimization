import numpy as np

from SimpleFEM.source.mesh import Mesh
from sources.LevelSetFunction import LevelSetFunction
from sources.fem_solution_interpolation import FemSolutionInterpolation


class FiniteDifference(LevelSetFunction):

    def __init__(
            self,
            mesh: Mesh,
            shape: tuple,
            level_set_vals: np.ndarray,
            space_delta: float,
    ):
        """
        :param mesh: mesh used by FEM on which velocity function will be computed
        :param shape: shape of mesh
        :param level_set_vals: signed distance function computed per center of element
        :param space_delta: distance between points on grid
        :param time_delta: discrete time step
        """
        self.grid_shape = np.array(shape)
        self.dh = space_delta

        # self.x_points = np.linspace(0, self.grid_shape[0], np.ceil(grid_shape[0] / space_delta))
        # self.y_points = np.linspace(0, self.grid_shape[1], np.ceil(segrid_shape[1] / space_delta))
        # self.phi = np.zeros(shape=(self.x_points.shape[0], self.y_points.shape[0]))

        dshape = tuple(int(np.floor(shape[i] / space_delta)) for i in (0, 1))
        self.phi = np.empty(dshape)
        print(self.phi.shape)

        self.fem_interpolator = FemSolutionInterpolation(mesh=mesh)
        self.compute_initial_condition(mesh=mesh, init_vals=level_set_vals)

    def __call__(self, point, *args, **kwargs):
        # todo interpolate this values
        idx = np.floor(point / self.dh).astype(int)
        return self.phi[idx[0], idx[1]]

    def compute_initial_condition(self, mesh: Mesh, init_vals: np.ndarray):
        self.fem_interpolator.set_values_to_interpolate(init_vals)
        for x in range(self.phi.shape[0]):
            for y in range(self.phi.shape[1]):
                point = np.array([x, y]) * self.dh
                self.phi[x, y] = self.fem_interpolator(point)


    # def compute_initial_condition(self):
    #     # todo compute \phi value at every grid point
    #     for x, vx in enumerate(self.x_points):
    #         for y, vy in enumerate(self.y_points):
    #             self.phi[x, y] = self.fem_interpolator(np.array([vx, vy]))
    #
    # def compute_velocity(self):
    #     pass

    def update(self, v_func: np.ndarray, dt: float):
        self.fem_interpolator.set_values_to_interpolate(v_func)
        new_phi = np.empty_like(self.phi)

        for x in range(self.phi.shape[0]):
            for y in range(self.phi.shape[1]):
                v = self.fem_interpolator(np.array([x, y]) * self.dh)
                new_phi[x, y] = self.new_phi_val(np.array([x, y]), v, dt)
        self.phi = new_phi

    def new_phi_val(self, idx: np.ndarray, v: float, dt: float):
        horizontal = np.array([1, 0])
        vertical = np.array([0, 1])

        explicit_x = self.explicit_scheme(idx, horizontal)
        implicit_x = self.implicit_scheme(idx, horizontal)

        explicit_y = self.explicit_scheme(idx, vertical)
        implicit_y = self.implicit_scheme(idx, vertical)

        max_v = np.max([0, v])
        min_v = np.min([0, v])

        if max_v == 0:
            return self.phi[idx[0], idx[1]] - dt * min_v * (self.g_minus(explicit_x, implicit_x) + self.g_minus(explicit_y, implicit_y))
        else:
            return self.phi[idx[0], idx[1]] - dt * max_v * (self.g_plus(explicit_x, implicit_x) + self.g_plus(explicit_x, implicit_x))

    def explicit_scheme(self, idx: np.ndarray, direction: np.ndarray):
        next_idx = idx + direction
        if any(next_idx >= self.grid_shape):
            return 0
        return (self.phi[next_idx[0], next_idx[1]] - self.phi[idx[0], idx[1]]) / self.dh

    def implicit_scheme(self, idx: np.ndarray, direction: np.ndarray):
        prev_idx = idx - direction
        if any(prev_idx < 0):
            return 0
        return (self.phi[idx[0], idx[1]] - self.phi[prev_idx[0], prev_idx[1]]) / self.dh

    def g_plus(self, d_plus, d_minus):
        return np.sqrt(np.min(d_plus, 0)**2 + np.max(d_minus, 0)**2)

    def g_minus(self, d_plus, d_minus):
        return np.sqrt(np.max(d_plus, 0)**2 + np.min(d_minus, 0)**2)
