import numpy as np
from matplotlib import pyplot as plt

from sources.level_set_function import LevelSetFunction
from sources.radial_base_functions import RadialBaseFunctions


class RbfFilterPoints(LevelSetFunction):

    def __init__(self, points_ratio: float, points: np.ndarray, init_values: np.ndarray, dims: int = 2, const_bias: float = 0.001):
        self.mask = np.random.choice([False, True], size=points.shape[0], p=[1 - points_ratio, points_ratio])

        plt.scatter(points[self.mask, 0], points[self.mask, 1])
        plt.title('rbfs centers')
        plt.show()

        self.rbf = RadialBaseFunctions(points[self.mask], init_values[self.mask], dims, const_bias)

    def __call__(self, x: np.ndarray, *args, **kwargs) -> float:
        return self.rbf(x)

    def update(self, velocity_values: np.ndarray, dt: float) -> None:
        self.rbf.update(velocity_values[self.mask], dt)

    def reinitialize(self, values: np.ndarray) -> None:
        self.rbf.reinitialize(values[self.mask])
