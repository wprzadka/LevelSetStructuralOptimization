import os
import sys
import numpy as np

# add SimpleFEM root directory to path in order to make relative imports works
FEM_PATH = os.path.abspath("SimpleFEM")
sys.path.append(FEM_PATH)

from SimpleFEM.source.mesh import Mesh
from sources.optimization import LevelSetMethod

from SimpleFEM.source.examples.materials import MaterialProperty

if __name__ == '__main__':
    mesh_path =  os.path.join(os.path.dirname(__file__), 'meshes/rectangle180x60v2.msh')

    mesh = Mesh(mesh_path)
    shape = (180, 60)

    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['right-down'])

    optim = LevelSetMethod(
        mesh,
        shape,
        MaterialProperty.Polystyrene,
        rhs_func=lambda x: np.array([0, 0]),
        dirichlet_func=lambda x: np.array([0, 0]),
        neumann_func=lambda x: np.array([0, -1e4])
    )
    optim.optimize(100)
