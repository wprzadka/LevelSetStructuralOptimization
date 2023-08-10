from itertools import combinations
import numpy as np

from SimpleFEM.source.mesh import Mesh


def construct_nodes_adj_graph(mesh: Mesh):
    neighbours = [set() for _ in range(mesh.nodes_num)]
    for nodes in mesh.nodes_of_elem:
        for beg, end in combinations(nodes, 2):
            neighbours[beg].add(end)
            neighbours[end].add(beg)
    neighbours = [list(v) for v in neighbours]
    assert all([len(x) > 0 for x in neighbours])
    return neighbours


def construct_elems_adj_graph(mesh: Mesh, elems_of_node:np.ndarray = None):
    if elems_of_node is None:
        elems_of_node = create_nodes_to_elems_mapping(mesh)
    neighbours = [set() for _ in range(mesh.elems_num)]
    for fst, nodes in enumerate(mesh.nodes_of_elem):
        close_elems = (elems_of_node[n] for n in nodes)
        for snd in (x for others in close_elems for x in others):
            if len([
                x for x in mesh.nodes_of_elem[fst]
                if x in mesh.nodes_of_elem[snd]
            ]) == 2:
                neighbours[fst].add(snd)
    neighbours = [list(x) for x in neighbours]
    assert all([len(x) > 0 for x in neighbours])
    return neighbours


def create_nodes_to_elems_mapping(mesh: Mesh):
    node_to_elems = [[] for _ in range(mesh.nodes_num)]
    for elem_idx, nodes in enumerate(mesh.nodes_of_elem):
        for n in nodes:
            node_to_elems[n].append(elem_idx)
    return node_to_elems
