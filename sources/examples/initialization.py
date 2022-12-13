import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import tri

# add SimpleFEM root directory to path in order to make relative imports works
FEM_PATH = os.path.abspath("SimpleFEM")
sys.path.append(FEM_PATH)

from SimpleFEM.source.mesh import Mesh
from sources.optimization import LevelSetMethod

from SimpleFEM.source.examples.materials import MaterialProperty
from sources.signed_distance import SignedDistanceInitialization

if __name__ == '__main__':

    # mesh_path = 'SimpleFEM/meshes/rectangle.msh'
    mesh_path = 'sources/examples/truss.msh'
    mesh = Mesh(mesh_path)
    # shape = (1., 0.5)
    shape = (180, 60)

    optim = LevelSetMethod(
        mesh,
        shape,
        MaterialProperty.Polystyrene,
        rhs_func=lambda x: np.array([0, 0]),
        dirichlet_func=lambda x: np.array([0, 0]),
        neumann_func=lambda x: np.array([0, -1e6])
    )
    sign_dist = SignedDistanceInitialization(domain_type='mesh', domain=mesh)

    density = optim.fill_uniformly_with_holes(holes_per_axis=(6, 3), radius=5)

    triangulation = tri.Triangulation(
        x=mesh.coordinates2D[:, 0],
        y=mesh.coordinates2D[:, 1],
        triangles=mesh.nodes_of_elem
    )
    plt.tripcolor(triangulation, density, cmap='gray_r')
    # plt.triplot(triangulation)
    plt.show()

    sign_distance = sign_dist(density)

    plt.tripcolor(triangulation, density, cmap='gray_r')
    plt.tricontour(triangulation, sign_distance)
    plt.show()

    fig = plt.figure(figsize=plt.figaspect(1))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot_trisurf(triangulation, sign_distance, cmap='seismic')
    plt.show()
