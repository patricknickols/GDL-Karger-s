import random

def number_of_nodes(edges):
    node_set = set()
    for source, target in edges:
        node_set.add(source)
        node_set.add(target)
    return len(node_set)

def midpoint(edge):
    source, target = edge
    return source + target / 2

def contract(edges, edge):
    source, target = edge
    new_node_label = (source + target)/2
    edges = [edge for edge in edges if midpoint(edge) != new_node_label]
    new_edges = []
    for other_source, other_target in edges: 
        if other_source in {source, target}:
            new_edges.append((new_node_label, other_target))
        elif other_target in {source, target}:
            new_edges.append((other_source, new_node_label))
        else:
            new_edges.append((other_source, other_target))
    return new_edges

        
def _karger(edges):
    while number_of_nodes(edges) > 2:
        e = random.choice(edges)
        edges = contract(edges, e)
    return edges