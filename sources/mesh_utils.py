from itertools import combinations

from SimpleFEM.source.mesh import Mesh


def construct_graph(mesh: Mesh):
    neighbours = [set() for _ in range(mesh.nodes_num)]
    for nodes in mesh.nodes_of_elem:
        for beg, end in combinations(nodes, 2):
            neighbours[beg].add(end)
            neighbours[end].add(beg)
    neighbours = [list(v) for v in neighbours]
    return neighbours


def create_nodes_to_elems_mapping(mesh: Mesh):
    node_to_elems = [[] for _ in range(mesh.nodes_num)]
    for elem_idx, nodes in enumerate(mesh.nodes_of_elem):
        for n in nodes:
            node_to_elems[n].append(elem_idx)
    return node_to_elems
