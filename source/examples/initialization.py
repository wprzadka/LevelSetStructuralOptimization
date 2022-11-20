import matplotlib.pyplot as plt
from matplotlib import tri

from SimpleFEM.source.mesh import Mesh
from source.optimization import LevelSetMethod


if __name__ == '__main__':

    mesh_path = 'SimpleFEM/meshes/rectangle.msh'
    mesh = Mesh(mesh_path)
    shape = (1., 0.5)

    optim = LevelSetMethod(mesh, shape)

    density = optim.fill_uniformly_with_holes(holes_per_axis=(6, 3), radius=0.04)

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, density, cmap='gray_r')
    plt.triplot(triangulation)
    plt.show()

    sign_distance = optim.compute_sign_distance(density)

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, density, cmap='gray_r')
    plt.tricontour(triangulation, sign_distance)
    plt.show()
