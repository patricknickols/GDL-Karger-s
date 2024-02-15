import random

def get_edges_from_dictionary(adjacency_dictionary):
    edges = []
    for key in adjacency_dictionary:
        for item in adjacency_dictionary[key]:
            if (item, key) not in edges:
                edges.append((key, item)) 
    return edges

def get_edges_from_adj_matrix(adj):
    edges = []
    for i in range(len(adj)):
        for j in range(len(adj[0])):
            if adj[i][j] != 0:
                edges.append((i,j)) 
    return edges

def contract(edges, edge):
    source = edge[0]
    target = edge[1]
    new_node_label = f"{source},{target}"
    edges = [edge for edge in edges if edge != (source, target)]
    edges = [edge for edge in edges if edge != (target, source)]
    new_edges = []
    for other_source, other_target in edges: 
        if other_source in {source, target}:
            new_edges.append((new_node_label, other_target))
        elif other_target in {source, target}:
            new_edges.append((other_source, new_node_label))
        else:
            new_edges.append((other_source, other_target))
    return new_edges

def number_of_nodes(edges):
    node_set = set()
    for source, target in edges:
        node_set.add(source)
        node_set.add(target)
    return len(node_set)
        
def _karger(edges):
    while number_of_nodes(edges) > 2:
        e = random.choice(edges)
        edges = contract(edges, e)
    return list(edges)


def karger_dict(adjacency_dictionary):
    edges = get_edges_from_dictionary(adjacency_dictionary)
    return _karger(edges)
    
def karger_mat(adjacency_matrix):
    edges = get_edges_from_adj_matrix(adjacency_matrix)
    return _karger(edges)
