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
    np.random.seed(0)
    mesh_path =  os.path.join(os.path.dirname(__file__), 'meshes/hole120x60v4.msh')

    mesh = Mesh(mesh_path)
    shape = (120, 60)

    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['right-down'])

    optim = LevelSetMethod(
        mesh=mesh,
        mesh_shape=shape,
        material=MaterialProperty.TestMaterial,
        rhs_func=lambda x: np.array([0, 0]),
        dirichlet_func=lambda x: np.array([0, 0]),
        neumann_func=lambda x: np.array([0, -1]),
        lag_mult=0.035,
        reinitialization_period=5,
        updates_num=15,
        holes_per_axis=(5, 5),
        holes_radius=0.6
    )
    optim.optimize(100)
