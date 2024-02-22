from random import random
from typing import Tuple

import chex
import networkx as nx
from clrs._src import probing
from clrs._src import specs
import numpy as np

_Array = np.ndarray
_Out = Tuple[_Array, probing.ProbesDict]
_OutputClass = specs.OutputClass


def sample_indices_with_weights(A):
    A_normalized = A / np.sum(A)

    flat_indices = np.arange(A.size).reshape(-1)
    probabilities = A_normalized.reshape(-1)

    sampled_index = np.random.choice(flat_indices, p=probabilities)

    i, j = np.unravel_index(sampled_index, A.shape)

    return i, j


def replace_edges(adj_matrix, i, j):
    adj_matrix[i, :] += adj_matrix[j, :]
    adj_matrix[:, i] += adj_matrix[:, j]

    adj_matrix[j, :] = 0
    adj_matrix[:, j] = 0

    adj_matrix[i, j] = 0
    adj_matrix[j, i] = 0

    return adj_matrix


def karger_gen(A: _Array, Seed: int) -> _Out:
    chex.assert_rank(A, 2)
    probes = probing.initialize(specs.SPECS['karger'])

    A_pos = np.arange(A.shape[0])
    print(100)
    probing.push(
        probes,
        specs.Stage.INPUT,
        next_probe={
            'pos': np.copy(A_pos) * 1.0 / A.shape[0],
            'A': np.copy(A),
            'adj': probing.graph(np.copy(A)),
            'seed': Seed
        })
    random.seed(Seed)
    group = np.arange(A.shape[0])
    graph_comp = np.copy(A)
    for s in range(A.shape[0] - 2):
        probing.push(
            probes,
            specs.Stage.HINT,
            next_probe={
                'group_h': np.copy(group),
                'graph_comp': np.copy(graph_comp),
            })

        i, j = sample_indices_with_weights(graph_comp)
        i = group[i]
        j = group[j]
        if A_pos[i] > A_pos[j]:
            tmp = i
            i = j
            j = tmp
        group[group == j] = i
        replace_edges(graph_comp, i, j)

    probing.push(probes, specs.Stage.OUTPUT, next_probe={'group': np.copy(group)})
    probing.finalize(probes)
    return group, probes


if __name__ == '__main__':
    G = nx.Graph()
    G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    print(nx.adjacency_matrix(G).toarray())
    print(karger_gen(nx.adjacency_matrix(G).toarray() * 1.0, 42))
