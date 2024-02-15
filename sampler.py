import networkx as nx
from clrs._src.algorithms import samplers

class ConnectedGraphSampler(samplers.Sampler):
    """Generates an E-R random connected graph of a specific size"""
    def _sample_data(self,
                     size: int,
                     p: float,
                     tries: int
                    ):
        for _ in range(tries):
            graph = self._random_er_graph(size, p)
            if nx.is_connected(graph):
                return graph
        raise Exception("Tries exceeded, maybe adjust p higher?")