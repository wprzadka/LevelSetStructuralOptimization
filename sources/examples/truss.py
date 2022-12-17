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

    mesh_path = 'SimpleFEM/meshes/rectangle.msh'
    # mesh_path = 'sources/examples/truss.msh'
    mesh = Mesh(mesh_path)
    shape = (1., 0.5)
    # shape = (180, 60)


    optim = LevelSetMethod(
        mesh,
        shape,
        MaterialProperty.Polystyrene,
        rhs_func=lambda x: np.array([0, 0]),
        dirichlet_func=lambda x: np.array([0, 0]),
        neumann_func=lambda x: np.array([0, -1e6])
    )
    optim.optimize(100)
