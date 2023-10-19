import math
from collections import defaultdict
import networkx as nx
import numpy as np
import pandas as pd

from ._base import GraphScorer


class CannistraciHebbScorer(GraphScorer):
    def __init__(self, input_network, path_length, method):
        super().__init__(input_network)
        self.path_length = path_length
        assert method in ["RA", "CH1", "CH2", "CH3"]
        self.method = method
        self.locality_dict = defaultdict(set)
        self.path_dict = defaultdict(list)
        self.degree_interior_dict = defaultdict(None)
        self.degree_exterior_dict = defaultdict(None)

    def fit(self, X, y=None):
        X = self.make_dataset(X)
        for e_pair in X.itertuples(name=None, index=False):
            self.update_paths_locality(e_pair[0], e_pair[1])
        return self

    def transform(self, X: pd.DataFrame):
        X = self.make_dataset(X)
        scores = []
        for e_pair in X.itertuples(index=False, name=None):
            if self.method == "RA":
                scores.append(self.resource_allocation(e_pair[0], e_pair[1]))
            elif self.method == "CH1":
                scores.append(self.ch1(e_pair[0], e_pair[1]))
            elif self.method == "CH2":
                scores.append(self.ch2(e_pair[0], e_pair[1]))
            elif self.method == "CH3":
                scores.append(self.ch3(e_pair[0], e_pair[1]))
        return np.array(scores).reshape(-1, 1)

    def update_paths_locality(self, u, v):
        if self.locality_dict.get((u, v)) is not None and self.path_dict.get((u, v)) is not None:
            return
        for path in nx.all_simple_paths(self.input_network, u, v, cutoff=self.path_length):
            if (len(path) - 1) != self.path_length:
                continue
            self.path_dict[(u, v)].append(path[1:-1])
            for node in path[1:-1]:
                self.locality_dict[(u, v)].add(node)

    def resource_allocation(self, u, v):
        score = 0
        for path in self.path_dict.get((u, v)):
            score += 1 / math.pow(
                math.prod([self.input_network.degree[i] for i in path]), 1 / (self.path_length - 1)
            )
        return score

    def ch1(self, u, v):
        score = 0
        for path in self.path_dict.get((u, v)):
            score += math.pow(
                math.prod([self.count_interior(u, v, i) for i in path]), 1 / (self.path_length - 1)
            ) / math.pow(
                math.prod([self.input_network.degree[i] for i in path]), 1 / (self.path_length - 1)
            )
        return score

    def ch2(self, u, v):
        score = 0
        for path in self.path_dict.get((u, v)):
            score += math.pow(
                math.prod([1 + self.count_interior(u, v, i) for i in path]),
                1 / (self.path_length - 1),
            ) / math.pow(
                math.prod([1 + self.count_exterior(u, v, i) for i in path]),
                1 / (self.path_length - 1),
            )
        return score

    def ch3(self, u, v):
        score = 0
        for path in self.path_dict.get((u, v)):
            score += 1 / math.pow(
                math.prod([1 + self.count_exterior(u, v, i) for i in path]),
                1 / (self.path_length - 1),
            )
        return score

    def count_interior(self, u, v, node):
        if self.degree_interior_dict.get((u, v, node)) is None:
            node_neighbors = set(
                i for i in nx.all_neighbors(self.input_network, node) if i not in [u, v]
            )
            node_degree_i = len(self.locality_dict.get((u, v)).intersection(node_neighbors))
            self.degree_interior_dict.update({(u, v, node): node_degree_i})
        return self.degree_interior_dict.get((u, v, node))

    def count_exterior(self, u, v, node):
        if self.degree_exterior_dict.get((u, v, node)) is None:
            node_neighbors = set(
                i for i in nx.all_neighbors(self.input_network, node) if i not in [u, v]
            )
            node_degree_e = len(node_neighbors.difference(self.locality_dict.get((u, v))))
            self.degree_exterior_dict.update({(u, v, node): node_degree_e})
        return self.degree_exterior_dict.get((u, v, node))
