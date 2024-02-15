import networkx as nx
from Karger import karger_dict
from copy import deepcopy
tries = 100

def random_connected_graph(n):
    for _ in range(10000):
        G = nx.newman_watts_strogatz_graph(n, 3, 0.8)
        if nx.is_connected(G):
            return G
    raise Exception("Tries Exceeded")

def expected_success_count(tries, size_of_graph):
    return (2 * tries) / ((size_of_graph) * (size_of_graph - 1))

def get_minimal_partition(graph):
    edge_set = nx.minimum_edge_cut(graph)
    graph_copy = deepcopy(graph)
    for source, target in edge_set:
        graph_copy.remove_edge(source, target)
    return list(nx.connected_components(graph_copy))

def get_cardinality_min_cut(graph):
    return len(nx.minimum_edge_cut(graph))

def test_graph(graph, tries):
    count = 0
    min_partition = get_minimal_partition(graph)
    for _ in range(tries):
        karger_result = karger_dict(graph)
        if len(karger_result) == get_cardinality_min_cut(graph):
            count += 1
        elif len(karger_result) < get_cardinality_min_cut(graph):
            print("ERROR!")
            print(f"{len(karger_result)=}")
            print(f"{get_cardinality_min_cut(small_graph)=}")
            print(f"{min_partition=}")
            print(f"{graph.edges=}")
            print(f"{karger_result=}")
            raise Exception("Bug detected.")
    e = expected_success_count(tries, 5)
    print(f"{e=}")
    print(f"{count=}")



small_graph = random_connected_graph(10)
medium_test_graph = random_connected_graph(100)
large_test_graph = random_connected_graph(1000)


test_graph(small_graph, 10000)
test_graph(medium_test_graph, 1000)
test_graph(large_test_graph, 100)


# TODO: the generated graphs almost always have minimum cuts of lengths (1, n-1)