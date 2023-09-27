import networkx as nx
import numpy as np
import pandas as pd

from ._base import GraphScorer


class CommonNeighborsScorer(GraphScorer):
    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        common_neighbors = []
        for row in X.itertuples():
            cn = nx.common_neighbors(self.input_network, row.node_i, row.node_j)
            common_neighbors.append(len(list(cn)))
        return np.array(common_neighbors).reshape(-1, 1)


class AdamicAdarScorer(GraphScorer):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        aa = nx.adamic_adar_index(self.input_network, X.itertuples(index=False, name=None))
        return np.array([i[-1] for i in aa]).reshape(-1, 1)


class ShortestPathScorer(GraphScorer):
    def __init__(self, input_network):
        super(ShortestPathScorer, self).__init__(input_network)
        self.shortest_path_mat = None

    def fit(self, X, y=None):
        self.shortest_path_mat = dict(nx.shortest_path_length(self.input_network))
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        sp = []
        for row in X.itertuples():
            if row.node_j in self.shortest_path_mat[row.node_i].keys():
                path_length = self.shortest_path_mat[row.node_i][row.node_j]
            else:
                # path_length =np.inf
                path_length = 9999
            sp.append(path_length)
        return np.array(sp).reshape(-1, 1)


class JaccardScorer(GraphScorer):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        js = nx.jaccard_coefficient(self.input_network, X.itertuples(index=False, name=None))
        return np.array([i[-1] for i in js]).reshape(-1, 1)


class PreferentialAttachmentScorer(GraphScorer):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        pa = nx.preferential_attachment(self.input_network, X.itertuples(index=False, name=None))
        return np.array([i[-1] for i in pa]).reshape(-1, 1)


class LHNScorer(GraphScorer):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        lhn = []
        for e_pair in X.itertuples(index=False, name=None):
            num_common_neighbors = len(
                list(nx.common_neighbors(self.input_network, e_pair[0], e_pair[1]))
            )
            degree_product = self.input_network.degree(e_pair[0]) * self.input_network.degree(
                e_pair[1]
            )
            if num_common_neighbors == 0 and degree_product == 0:
                lhn.append(0)
            else:
                lhn.append(num_common_neighbors / degree_product)
        return np.array(lhn).reshape(-1, 1)


class PersonalizedPageRankScorer(GraphScorer):
    def __init__(self, input_network):
        super(PersonalizedPageRankScorer, self).__init__(input_network)
        self.pers_page_rank = {}

    def fit(self, X, y=None):
        num_nodes = nx.number_of_nodes(self.input_network)
        hot_vec = dict(zip(range(num_nodes), [0] * num_nodes))
        for node in range(num_nodes):
            hot_vec.update({node: 1})
            self.pers_page_rank[node] = nx.pagerank(self.input_network, personalization=hot_vec)
            hot_vec.update({node: 0})
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        ppr = []
        for e_pair in X.itertuples(index=False, name=None):
            ppr.append(self.pers_page_rank[e_pair[0]][e_pair[1]])
        return np.array(ppr).reshape(-1, 1)
