import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import center_of_mass
from sources.LevelSetFunction import LevelSetFunction


class RadialBaseFunctions(LevelSetFunction):

    def __init__(self, points: np.ndarray, init_values: np.ndarray, dims: int = 2, const_bias: float = 0.001):
        self.dims = dims
        self.bias = const_bias
        self.points = points
        self.points_num = points.shape[0]
        self.h_matrix = self.create_h_matrix()
        self.inv_h_matrix = np.linalg.inv(self.h_matrix)

        self.reinitialize(init_values)
        self.grad_in_points = self.create_grad_in_points()

    def reinitialize(self, values: np.ndarray):
        rhs = np.concatenate((
            values,
            np.zeros(shape=(self.dims + 1))
        ))
        self.coefficients = self.inv_h_matrix @ rhs
        if __debug__:
            self.plot_surface()

    # TODO vectorise
    def __call__(self, x: np.ndarray):
        diffs = x - self.points
        rbf_vals = self.rbf(diffs)
        vals = np.concatenate((rbf_vals, np.ones(1), x))
        return vals @ self.coefficients

    def rbf(self, diff: np.ndarray):
        return np.sqrt(np.sum(diff ** 2, axis=-1) + self.bias ** 2)

    def grad(self, diff: np.ndarray):
        return diff / np.sqrt(np.sum(diff ** 2, axis=-1) + self.bias ** 2)[:, :, np.newaxis]

    def update(self, velocities: np.ndarray, dt: float):
        fst = self.grad_in_points[:, :, 0] @ self.coefficients
        snd = self.grad_in_points[:, :, 1] @ self.coefficients

        b = np.empty(velocities.size + self.dims + 1)
        b[:velocities.size] = velocities * np.sqrt(fst ** 2 + snd ** 2)
        b[velocities.size:] = np.zeros(self.dims + 1)

        change = dt * self.inv_h_matrix @ b
        self.coefficients -= change

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
        grad_in_points = np.empty((self.points_num, size, self.dims))
        # todo check self.points[:, None, :] or self.points[None, :]
        diffs = self.points[:, None, :] - self.points
        grad_in_points[:self.points_num, :self.points_num] = self.grad(diffs)
        grad_in_points[:, self.points_num:] = np.array([[0, 0], [1, 0], [0, 1]])
        return grad_in_points

    def plot_surface(self):
        dom_x = np.linspace(0, 180, 100)
        dom_y = np.linspace(0, 60, 100)
        X, Y = np.meshgrid(dom_x, dom_y)
        Z = np.array([
            self.__call__(v) for v in zip(X.flatten(), Y.flatten())
        ]).reshape(X.shape)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_box_aspect((np.ptp(X), np.ptp(Y), 20))
        ax.plot_surface(X, Y, np.zeros_like(X), color='tab:blue')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        fig.show()
        plt.close(fig)


if __name__ == '__main__':
    func = lambda x: np.sum(np.sin(x/10), axis=1)
    mesh = Mesh('sources/examples/meshes/rectangle180x60v3.msh')
    points = np.array([center_of_mass(mesh.coordinates2D[nodes]) for nodes in mesh.nodes_of_elem])
    exact = func(points)

    rbf_interpolation = RadialBaseFunctions(points, exact, dims=2)

    values = np.array([rbf_interpolation(x) for x in points])

    plt.imshow(rbf_interpolation.h_matrix)
    plt.show()

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, exact)
    plt.colorbar()
    plt.show()

    plt.tripcolor(triangulation, values)
    plt.colorbar()
    fst = rbf_interpolation.grad_in_points[:, :, 0] @ rbf_interpolation.coefficients
    snd = rbf_interpolation.grad_in_points[:, :, 1] @ rbf_interpolation.coefficients
    plt.quiver(points[:,0], points[:,1], fst, snd)
    plt.show()

    dom_x = np.linspace(0, 180, 100)
    dom_y = np.linspace(0, 60, 100)
    X, Y = np.meshgrid(dom_x, dom_y)
    Z = np.array([
        rbf_interpolation(v) for v in zip(X.flatten(), Y.flatten())
    ]).reshape(X.shape)
    plt.matshow(Z)
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.show()
