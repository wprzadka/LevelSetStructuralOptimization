import numpy as np
from matplotlib import pyplot as plt, tri

from SimpleFEM.source.mesh import Mesh


def generate_cosine_func(shape: tuple, points: np.ndarray, holes_per_axis: tuple, radius: float):
    phi = -np.cos(points[:, 0] / shape[0] * holes_per_axis[0] * np.pi) * np.cos(points[:, 1] / shape[1] * holes_per_axis[1] * np.pi) + radius - 1
    return phi


if __name__ == '__main__':
    mesh = Mesh('sources/examples/meshes/rectangle180x60v3.msh')
    phi = generate_cosine_func((180, 60), mesh, (8, 4), 0.6)

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tricontour(triangulation, phi)
    plt.show()
