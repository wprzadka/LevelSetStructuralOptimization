import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from SimpleFEM.source.mesh import Mesh


class RadialBaseFunctions:

    def __init__(self, mesh: Mesh, init_values: np.ndarray, dims: int = 2):
        self.dims = dims
        self.points = mesh.coordinates2D
        self.points_num = self.points.shape[0]

        self.h_matrix = self.create_h_matrix()
        self.inv_h_matrix = np.linalg.inv(self.h_matrix)

        rhs = np.concatenate((
            init_values,
            np.zeros(shape=(self.dims + 1))
        ))
        self.coefficients = self.inv_h_matrix @ rhs

    def __call__(self, x: np.ndarray):
        diffs = x - self.points
        rbf_vals = self.rbf(diffs)
        vals = np.concatenate((rbf_vals, np.ones(1), x))
        return vals @ self.coefficients

    def rbf(self, diff: np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=-1))

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
