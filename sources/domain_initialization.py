import numpy as np
from matplotlib import pyplot as plt, tri
from skfmm import distance
from scipy.interpolate import RegularGridInterpolator, griddata

from SimpleFEM.source.mesh import Mesh

def reinitalize(shape: tuple, points: np.ndarray, phi: np.ndarray):
    dom_x = np.linspace(0, shape[0], shape[0])
    dom_y = np.linspace(0, shape[1], shape[1])
    grid = [(x, y) for x in dom_x for y in dom_y]

    values = griddata(points, phi, grid).reshape(shape)
    signed_dist = distance(values)

    interp = RegularGridInterpolator((dom_x, dom_y), signed_dist)
    return interp(points)

def initialize_sign_distance(shape: tuple, points: np.ndarray, holes_per_axis: tuple, radius: float):
    dom_x = np.linspace(0, 1, shape[0])
    dom_y = np.linspace(0, 1, shape[1])
    X, Y = np.meshgrid(dom_x, dom_y)
    base_vals = -np.cos(X * holes_per_axis[0] * np.pi) * np.cos(Y * holes_per_axis[1] * np.pi) + radius - 1
    domain = np.where(base_vals > 0, 1, -1)
    signed_dist = distance(domain)

    interp = RegularGridInterpolator((dom_x * shape[0], dom_y * shape[1]), signed_dist.T)
    return interp(points)


def generate_cosine_func(shape: tuple, points: np.ndarray, holes_per_axis: tuple, radius: float):
    phi = -np.cos(points[:, 0] / shape[0] * holes_per_axis[0] * np.pi) * np.cos(points[:, 1] / shape[1] * holes_per_axis[1] * np.pi) + radius - 1
    return phi


if __name__ == '__main__':
    mesh = Mesh('sources/examples/meshes/rectangle180x60v3.msh')
    phi = generate_cosine_func((180, 60), mesh.coordinates2D, (4, 2), 0.6)

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, phi)
    plt.show()

    sign_dist_reinit = reinitalize((180, 60), mesh.coordinates2D, phi)
    plt.tripcolor(triangulation, sign_dist_reinit)
    plt.show()

    sign_dist = initialize_sign_distance((180, 60), mesh.coordinates2D, (4, 2), 0.6)
    plt.tripcolor(triangulation, sign_dist)
    plt.show()
