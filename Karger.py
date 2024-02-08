import random
def get_edges_from_dictionary(adjacency_dictionary):
    edges = set()
    for key in adjacency_dictionary:
        for item in adjacency_dictionary[key]:
            if (item,key) not in edges:
                edges.add((key, item)) 
    return edges

def get_edges_from_adj_matrix(adj):
    return [(i,j) for i in range(len(adj) for j in range(len(adj[0]))) if adj[i][j] != 0]


def contract(edges, edge):
    source = edge[0]
    target = edge[1]
    new_node_label = f"{source},{target}"
    edges.discard((source, target))
    edges.discard((target, source))
    removals = set()
    additions = set()
    for other_source, other_target in edges: 
        if other_source in {source, target}:
            removals.add((other_source, other_target))
            additions.add((new_node_label, other_target))
        elif other_target in {source, target}:
            removals.add((other_source, other_target))
            additions.add((other_source, new_node_label))
    edges -= removals
    edges |= additions
    return edges

        

def number_of_nodes(edges):
    node_set = set()
    for source, target in edges:
        node_set.add(source)
        node_set.add(target)
    return len(node_set)
        
def _karger(edges):
    while number_of_nodes(edges) > 2:
        e = random.choice(list(edges))
        edges = contract(edges, e)
    return list(edges)[0]


def karger_dict(adjacency_dictionary):
    edges = get_edges_from_dictionary(adjacency_dictionary)
    return _karger(edges)
    
def karger_mat(adjacency_matrix):
    edges = get_edges_from_adj_matrix(adjacency_matrix)
    return _karger(edges)




test_graph_dictionary = {1: {2,4,5}, 
                         2: {1,3,4}, 
                         3: {2,4},
                         4: {1,2,3,5},
                         5: {1,4}
                        }

print(karger_dict(test_graph_dictionary))
