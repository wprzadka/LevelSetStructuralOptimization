import numpy as np


class MeshFixture:
    def __init__(self, coordinates2D: np.ndarray, nodes_of_elem: np.ndarray):
        self.coordinates2D = coordinates2D
        self.nodes_of_elem = nodes_of_elem

