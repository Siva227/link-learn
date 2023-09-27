import networkx as nx
import numpy as np

from ._base import GraphScorer


class PageRankScorer(GraphScorer):
    def __init__(self, input_network):
        super(PageRankScorer, self).__init__(input_network)
        self.page_rank_dict_ = None

    def fit(self, X, y=None):
        self.page_rank_dict_ = nx.pagerank(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        pr = [self.page_rank_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(pr).reshape(-1, 1)


class LocalClusteringCoefficientScorer(GraphScorer):
    def __init__(self, input_network):
        super(LocalClusteringCoefficientScorer, self).__init__(input_network)
        self.local_clustering_dict_ = None

    def fit(self, X, y=None):
        self.local_clustering_dict_ = nx.clustering(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        lcc = [self.local_clustering_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(lcc).reshape(-1, 1)


class EigenvectorCentralityScorer(GraphScorer):
    def __init__(self, input_network, tolerance=1e-6):
        super(EigenvectorCentralityScorer, self).__init__(input_network)
        self.tolerance = tolerance
        self.eig_cen_dict_ = None

    def fit(self, X, y=None):
        flag = 1
        while flag == 1:
            try:
                self.eig_cen_dict_ = nx.eigenvector_centrality(
                    self.input_network, tol=self.tolerance
                )
                flag = 0
            except:
                self.tolerance = self.tolerance * 1e1
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        eig_cent = [self.eig_cen_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(eig_cent).reshape(-1, 1)


class DegreeCentralityScorer(GraphScorer):
    def __init__(self, input_network):
        super(DegreeCentralityScorer, self).__init__(input_network)
        self.deg_cen_dict_ = None

    def fit(self, X, y=None):
        self.deg_cen_dict_ = nx.degree_centrality(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        deg_cen = [self.deg_cen_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(deg_cen).reshape(-1, 1)


class ClosenessCentralityScorer(GraphScorer):
    def __init__(self, input_network):
        super(ClosenessCentralityScorer, self).__init__(input_network)
        self.closeness_cent_dict_ = None

    def fit(self, X, y=None):
        self.closeness_cent_dict_ = nx.closeness_centrality(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        cc = [self.closeness_cent_dict_[row.node_i] for row in X.itertuples()]
        return np.array(cc).reshape(-1, 1)


class BetweennessCentralityScorer(GraphScorer):
    def __init__(self, input_network, normalized=True):
        super(BetweennessCentralityScorer, self).__init__(input_network)
        self.normalized = normalized
        self.bet_cen_dict_ = None

    def fit(self, X, y=None):
        self.bet_cen_dict_ = nx.betweenness_centrality(self.input_network, self.normalized)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        bet_cen = [self.bet_cen_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(bet_cen).reshape(-1, 1)


class LoadCentralityScorer(GraphScorer):
    def __init__(self, input_network, normalized=True):
        super(LoadCentralityScorer, self).__init__(input_network)
        self.normalized = normalized
        self.load_cen_dict_ = None

    def fit(self, X, y=None):
        self.load_cen_dict_ = nx.load_centrality(self.input_network, normalized=self.normalized)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        load_cen = [self.load_cen_dict_[row.node_i] for row in X.itertuples()]
        return np.array(load_cen).reshape(-1, 1)


class KatzCentralityScorer(GraphScorer):
    def __init__(self, input_network):
        super(KatzCentralityScorer, self).__init__(input_network)
        self.katz_cen_dict_ = None

    def fit(self, X, y=None):
        self.katz_cen_dict_ = nx.katz_centrality_numpy(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        katz_cen = [self.katz_cen_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(katz_cen).reshape(-1, 1)


class NumTrianglesScorer(GraphScorer):
    def __init__(self, input_network):
        super(NumTrianglesScorer, self).__init__(input_network)
        self.num_triangles_dict_ = None

    def fit(self, X, y=None):
        self.num_triangles_dict_ = nx.triangles(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        num_triangles = [self.num_triangles_dict_[row.node_i] for row in X.itertuples(index=False)]
        return np.array(num_triangles).reshape(-1, 1)


class AvgNeighborDegreeScorer(GraphScorer):
    def __init__(self, input_network):
        super(AvgNeighborDegreeScorer, self).__init__(input_network)
        self.avg_neighbor_degree_dict_ = None

    def fit(self, X, y=None):
        self.avg_neighbor_degree_dict_ = nx.average_neighbor_degree(self.input_network)
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        avg_neighbor_deg = [
            self.avg_neighbor_degree_dict_[row.node_i] for row in X.itertuples(index=False)
        ]
        return np.array(avg_neighbor_deg).reshape(-1, 1)
