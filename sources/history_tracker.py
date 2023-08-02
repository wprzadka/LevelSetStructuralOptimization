import numpy as np

from SimpleFEM.source.mesh import Mesh
from SimpleFEM.source.utilities.computation_utils import area_of_triangle


class HistoryTracker:

    def __init__(self, mesh: Mesh, lag_mult: float):
        self.elem_volumes = np.array([
            area_of_triangle(mesh.coordinates2D[nodes_ids])
            for nodes_ids in mesh.nodes_of_elem
        ])
        self.lag_mult = lag_mult
        self.history = {'cost': [], 'compliance': [], "weight": []}

    def log(self, density: np.ndarray, elems_compliance: np.ndarray):
        weight = np.sum(density * self.elem_volumes)
        compliance = np.sum(elems_compliance)
        cost = compliance + self.lag_mult * weight

        print(f'cost = {cost}')
        print(f'compliance = {compliance}')
        print(f'weight = {weight}')

        self.history['cost'].append(cost)
        self.history['compliance'].append(compliance)
        self.history['weight'].append(weight)
