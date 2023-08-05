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
    mesh_path =  os.path.join(os.path.dirname(__file__), 'meshes/rectangle120x60v4.msh')

    mesh = Mesh(mesh_path)
    shape = (120, 60)

    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['left'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['right'])

    optim = LevelSetMethod(
        mesh=mesh,
        mesh_shape=shape,
        material=MaterialProperty.TestMaterial,
        rhs_func=lambda x: np.array([0, 0]),
        dirichlet_func=lambda x: np.array([0, 0]),
        neumann_func=lambda x: np.array([0, -1]) if 28.5 < x[1] < 31.5 else np.zeros(2),
        lag_mult=1.,
        reinitialization_period=5,
        updates_num=10,
        holes_per_axis=(6, 2),
        holes_radius=0.8
    )
    optim.optimize(100)
