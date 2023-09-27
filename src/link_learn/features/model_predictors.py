import graph_tool as gt
import networkx as nx
import numpy as np
from community import community_louvain
from graph_tool.all import BlockState, minimize_blockmodel_dl
from infomap import Infomap

from ..utils import nx2gt
from ._base import GraphScorer

# TODO: Improve computation speed with parallelization where possible
# TODO: Add Documentation


class LouvainScorer(GraphScorer):
    def __init__(
        self,
        input_network: nx.Graph,
        partition=None,
        weight="weight",
        resolution=1.0,
        randomize=None,
        random_state=None,
    ):
        super(LouvainScorer, self).__init__(input_network)
        self.weight = weight
        self.partition = partition
        self.resolution = resolution
        self.randomize = randomize
        self.random_state = random_state
        self.best_partition_ = None
        self.base_modularity_ = None

    def fit(self, X, y=None):
        self.best_partition_ = community_louvain.best_partition(
            self.input_network,
            self.partition,
            self.weight,
            self.resolution,
            self.randomize,
            self.random_state,
        )
        self.base_modularity_ = community_louvain.modularity(
            self.best_partition_, self.input_network
        )
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        score = []
        for e_pair in X.itertuples(name=None, index=False):
            # Commenting out the below code in the interest of efficiency.
            # Assumption is that the edges being scored are not part of the initial network
            # Which is true for our analysis.

            # in_copy = self.input_network.copy()
            # if e_pair in in_copy.edges:
            #     in_copy.remove_edge(*e_pair)
            # q_wo_edge = community_louvain.modularity(self.partition, in_copy)
            # in_copy.add_edge(*e_pair)
            # q_w_edge = community_louvain.modularity(self.partition, in_copy)
            # score.append(q_w_edge - q_wo_edge)
            q_wo_edge = self.base_modularity_
            self.input_network.add_edge(*e_pair)
            q_w_edge = community_louvain.modularity(self.best_partition_, self.input_network)
            self.input_network.remove_edge(*e_pair)
            score.append(q_w_edge - q_wo_edge)
        return np.array(score).reshape(-1, 1)


class InfomapScorer(GraphScorer):
    def __init__(self, input_network, args=None, two_level=True, num_trials=1):
        super(InfomapScorer, self).__init__(input_network)
        self.args = args
        self.two_level = two_level
        self.num_trials = num_trials
        self.im_ = None
        # self.im = Infomap(args=args, two_level=two_level, silent=True, num_trials=num_trials)
        # self.im.add_networkx_graph(self.input_network)
        self.im_modules_ = None
        self.im_code_length_ = None

    def fit(self, X, y=None):
        self.im_ = Infomap(
            args=self.args, two_level=self.two_level, silent=True, num_trials=self.num_trials
        )
        self.im_.add_networkx_graph(self.input_network)
        self.im_.run()
        self.im_modules_ = self.im_.get_modules()
        self.im_code_length_ = self.im_.codelength
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        im_score = []
        for e_pair in X.itertuples(name=None, index=False):
            # Commenting out the below code in the interest of efficiency.
            # Assumption is that the edges being scored are not part of the initial network
            # Which is true for our analysis.

            # if e_pair in self.input_network.edges:
            #     self.im.remove_link(*e_pair)
            #     self.im.run(initial_partition=self.im_modules, no_infomap=True)
            #     im_wo_edge = self.im.codelength
            # else:
            #     im_wo_edge = self.im_code_length
            im_wo_edge = self.im_code_length_
            self.im_.add_link(*e_pair)
            self.im_.run(initial_partition=self.im_modules_, no_infomap=True)
            im_w_edge = self.im_.codelength
            im_score.append(im_wo_edge - im_w_edge)
            self.im_.remove_link(*e_pair)
            # if e_pair not in self.input_network.edges:
            #     self.im.remove_link(*e_pair)
        return np.array(im_score).reshape(-1, 1)


class MDLScorer(GraphScorer):
    def __init__(self, input_network, deg_corr=False):
        super(MDLScorer, self).__init__(input_network)
        self.gt_in = nx2gt(input_network)
        self.deg_corr = deg_corr
        self.block_state_ = None
        self.base_entropy_ = None

    def fit(self, X, y=None):
        self.block_state_ = minimize_blockmodel_dl(
            self.gt_in, state_args=dict(deg_corr=self.deg_corr)
        )
        self.base_entropy_ = self.block_state_.entropy()
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        dl_score = [
            self.block_state_.get_edges_prob([i]) for i in X.itertuples(name=None, index=False)
        ]
        return np.array(dl_score).reshape(-1, 1)
