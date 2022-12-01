from typing import Tuple

import numpy as np
import matplotlib as plt


class FiniteDifference:

    def __init__(self, grid_size: Tuple[int, int], space_delta: float, time_delta: float):
        self.grid_size = grid_size
        self.dh = space_delta
        self.dt = time_delta

        x_points = np.linspace(0, grid_size[0], np.ceil(grid_size[0] / space_delta))
        y_points = np.linspace(0, grid_size[1], np.ceil(grid_size[1] / space_delta))
        self.phi = np.zeros(shape=(x_points.shape[0], y_points.shape[0]))

    def compute_initial_condition(self):
        # todo compute \phi value at every grid point
        pass

    def compute_velocity(self):
        pass

    def update(self, v: np.ndarray):
        new_phi = np.empty_like(self.phi)
        for y in range(self.phi.shape[0]):
            for x in range(self.phi.shape[1]):
                new_phi[x, y] = self.new_phi_val(np.array([x, y]), v)

    def new_phi_val(self, idxs: np.ndarray, v: np.ndarray):
        horizontal = np.array([1, 0])
        vertical = np.array([0, 1])

        explicit_x = self.explicit_scheme(idxs, horizontal)
        implicit_x = self.implicit_scheme(idxs, horizontal)

        explicit_y = self.explicit_scheme(idxs, vertical)
        implicit_y = self.implicit_scheme(idxs, vertical)

        return self.phi[idxs] \
               + np.max(0, v[idxs]) * self.g_plus(explicit_x, implicit_x) \
               + np.min(0, v[idxs]) * self.g_minus(explicit_x, implicit_x) \
               + np.max(0, v[idxs]) * self.g_plus(explicit_y, implicit_y) \
               + np.min(0, v[idxs]) * self.g_minus(explicit_y, implicit_y)

    def explicit_scheme(self, idxs: np.ndarray, direction: np.ndarray):
        return (self.phi[idxs + direction] - self.phi[idxs]) / self.dh

    def implicit_scheme(self, idxs: np.ndarray, direction: np.ndarray):
        return (self.phi[idxs] - self.phi[idxs - direction]) / self.dh

    def g_plus(self, d_plus, d_minus):
        return np.sqrt(np.min(d_plus, 0)**2 + np.max(d_minus, 0)**2)

    def g_minus(self, d_plus, d_minus):
        return np.sqrt(np.max(d_plus, 0)**2 + np.min(d_minus, 0)**2)
