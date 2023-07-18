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


def construct_elems_adj_graph(mesh: Mesh):
    neighbours = [[] for _ in range(mesh.elems_num)]
    for fst in range(mesh.elems_num):
        for snd in range(fst + 1, mesh.elems_num):
            if len([
                x for x in mesh.nodes_of_elem[fst]
                if x in mesh.nodes_of_elem[snd]
            ]) == 2:
                neighbours[fst].append(snd)
                neighbours[snd].append(fst)

    assert all([len(x) > 0 for x in neighbours])
    return neighbours


def create_nodes_to_elems_mapping(mesh: Mesh):
    node_to_elems = [[] for _ in range(mesh.nodes_num)]
    for elem_idx, nodes in enumerate(mesh.nodes_of_elem):
        for n in nodes:
            node_to_elems[n].append(elem_idx)
    return node_to_elems
