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
    mesh_path =  os.path.join(os.path.dirname(__file__), 'meshes/l_domain80x80v3.msh')

    mesh = Mesh(mesh_path)
    mesh.draw()
    shape = (80, 80)

    mesh.set_boundary_condition(Mesh.BoundaryConditionType.DIRICHLET, ['top'])
    mesh.set_boundary_condition(Mesh.BoundaryConditionType.NEUMANN, ['middle-right'])

    optim = LevelSetMethod(
        mesh=mesh,
        mesh_shape=shape,
        material=MaterialProperty.TestMaterial,
        rhs_func=lambda x: np.array([0, 0]),
        dirichlet_func=lambda x: np.array([0, 0]),
        neumann_func=lambda x: np.array([0, -1]),
        lag_mult=0.06,
        reinitialization_period=5,
        updates_num=15,
        holes_per_axis=(6, 6),
        holes_radius=0.7
    )
    optim.optimize(100)
