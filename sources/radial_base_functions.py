import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from SimpleFEM.source.mesh import Mesh


class RadialBaseFunctions:

    def __init__(self, mesh: Mesh, init_values: np.ndarray, dims: int = 2, const_bias: float = 0.001):
        self.dims = dims
        self.bias = const_bias

        self.points = mesh.coordinates2D
        self.points_num = self.points.shape[0]

        self.h_matrix = self.create_h_matrix()
        self.inv_h_matrix = np.linalg.inv(self.h_matrix)

        rhs = np.concatenate((
            init_values,
            np.zeros(shape=(self.dims + 1))
        ))
        self.coefficients = self.inv_h_matrix @ rhs
        self.grad_in_points = self.create_grad_in_points()

    def __call__(self, x: np.ndarray):
        diffs = x - self.points
        rbf_vals = self.rbf(diffs)
        vals = np.concatenate((rbf_vals, np.ones(1), x))
        return vals @ self.coefficients

    def rbf(self, diff: np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=-1) + self.bias)

    def grad(self, diff: np.ndarray):
        return diff / np.sqrt(np.sum(diff ** 2, axis=-1) + self.bias)[:, :, np.newaxis]

    def time_update(self, velocities: np.ndarray, dt: float):
        fst = self.grad_in_points[:, :, 0] @ self.coefficients
        snd = self.grad_in_points[:, :, 1] @ self.coefficients
        velocities = np.concatenate((velocities, np.zeros(self.dims + 1)))
        b = velocities * np.sqrt(fst ** 2 + snd ** 2)
        self.coefficients -= dt * self.inv_h_matrix @ b
        return self.coefficients

    def create_h_matrix(self):
        h_matrix_size = self.points_num + self.dims + 1
        h_matrix = np.zeros(shape=(h_matrix_size, h_matrix_size))
        # Compute A
        diffs = self.points[:, None, :] - self.points
        h_matrix[:self.points_num, :self.points_num] = self.rbf(diffs)
        # Compute P
        h_matrix[self.points_num, :self.points_num] = 1
        h_matrix[:self.points_num, self.points_num] = 1
        h_matrix[(self.points_num + 1):, :self.points_num] = self.points.T
        h_matrix[:self.points_num, (self.points_num + 1):] = self.points
        return h_matrix

    def create_grad_in_points(self):
        size = self.points_num + self.dims + 1
        grad_in_points = np.empty((size, size, self.dims))

        diffs = self.points[:, None, :] - self.points
        grad_in_points[:self.points_num, :self.points_num] = self.grad(diffs)

        grad_in_points[self.points_num, :] = np.zeros(self.dims)
        grad_in_points[:, self.points_num] = np.zeros(self.dims)
        for d in range(self.dims):
            grad = np.zeros(self.dims)
            grad[d] = 1
            grad_in_points[self.points_num, :] = grad
        return grad_in_points

if __name__ == '__main__':
    func = lambda x: np.sum(np.sin(x ** 2), axis=1)

    mesh = Mesh('SimpleFEM/meshes/rectangle.msh')

    exact = func(mesh.coordinates2D)
    rbf_interpolation = RadialBaseFunctions(mesh, exact, dims=2)

    values = np.array([rbf_interpolation(x) for x in mesh.coordinates2D])

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )

    plt.imshow(rbf_interpolation.h_matrix)
    plt.show()

    plt.tricontour(triangulation, exact, levels=50)
    plt.colorbar()
    plt.show()
    plt.tricontour(triangulation, values, levels=50)
    plt.colorbar()
    plt.show()
