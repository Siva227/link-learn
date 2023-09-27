import itertools
import warnings

import networkx as nx
import pandas as pd
from littleballoffur.edge_sampling import (
    HybridNodeEdgeSampler,
    RandomEdgeSampler,
    RandomEdgeSamplerWithInduction,
)
from sklearn.utils import check_random_state, shuffle


class GraphSampler:
    sampler_dict = {
        "rs": RandomEdgeSampler,
        "rswi": RandomEdgeSamplerWithInduction,
        "hnes": HybridNodeEdgeSampler,
    }

    def __init__(self, input_network, sampling_method="rs", alpha=0.8, alpha_=0.8, random_state=42):
        self.input_network = input_network
        self.sampling_method = sampling_method
        self.alpha = alpha
        self.alpha_ = alpha_
        self.random_state = random_state
        self.G_ho = nx.Graph()
        self.G_ho.add_nodes_from(self.input_network.nodes)
        self.G_tr = nx.Graph()
        self.G_tr.add_nodes_from(self.input_network.nodes)

    def sample(self, num_samples=10000, shuffle_flag=False):
        self.random_state = check_random_state(self.random_state)
        self.create_subgraphs()
        tr_df = self.get_pos_neg_edges(self.G_ho, self.G_tr)
        ho_df = self.get_pos_neg_edges(self.input_network, self.G_ho)

        tr_sample = tr_df.groupby("label").sample(
            n=num_samples, replace=True, random_state=self.random_state
        )
        ho_sample = ho_df.groupby("label").sample(
            n=num_samples, replace=True, random_state=self.random_state
        )
        if shuffle_flag:
            tr_sample = shuffle(tr_sample)
            ho_sample = shuffle(ho_sample)
            tr_sample.reset_index(drop=True, inplace=True)
            ho_sample.reset_index(drop=True, inplace=True)
        return tr_sample, ho_sample

    def create_subgraphs(self):
        n_edges_ho = int(self.alpha * nx.number_of_edges(self.input_network))
        s1 = self.sampler_dict[self.sampling_method](n_edges_ho)
        G1: nx.Graph = s1.sample(self.input_network)
        self.G_ho.add_edges_from(G1.edges)
        n_edges_tr = int(self.alpha_ * nx.number_of_edges(self.G_ho))
        s2 = self.sampler_dict[self.sampling_method](n_edges_tr)
        G2 = s2.sample(self.G_ho)
        orig_num_e = self.input_network.number_of_edges()
        ho_num_e = G1.number_of_edges()
        tr_num_e = G2.number_of_edges()
        assert tr_num_e < ho_num_e < orig_num_e, (
            f"Sampling failed: Expected edge counts\n(orig:{orig_num_e}, holdout:{n_edges_ho}, train:{n_edges_tr})\n"
            f"Found edge counts\n(orig:{orig_num_e}, holdout:{ho_num_e}, train:{tr_num_e})\n"
        )
        self.G_tr.add_edges_from(G2.edges)

    def get_pos_neg_edges(self, G_orig, G_sample):
        all_node_pairs = itertools.combinations(G_orig.nodes, 2)
        pos_edges = list(set(G_orig.edges).difference(set(G_sample.edges)))
        neg_edges = [i for i in all_node_pairs if i not in G_orig.edges]
        data = pos_edges + neg_edges
        label = [1] * len(pos_edges) + [0] * len(neg_edges)
        df = pd.DataFrame(data, columns=["node_i", "node_j"]).assign(label=label)
        return df
