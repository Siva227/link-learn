from collections import defaultdict

import networkx as nx
import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
from scipy.spatial.distance import cosine, euclidean
from tensorflow import keras
from tensorflow.keras import layers

from ._base import GraphScorer


class DeepWalkScorer(GraphScorer):
    metric_dict = {"euclidean": euclidean, "cosine": cosine}

    def __init__(
        self,
        input_network,
        num_walks=10,
        walk_length=80,
        embedding_size=128,
        window_size=10,
        num_epochs=10,
        learning_rate=0.01,
        metric="euclidean",
    ):
        super(DeepWalkScorer, self).__init__(input_network)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.adjacency_matrix = None
        self.embedding_matrix = None
        self.metric = metric

    def fit(self, X, y=None):
        self.adjacency_matrix = nx.to_numpy_array(self.input_network)
        self.num_nodes = self.adjacency_matrix.shape[0]
        self.node_indices = np.arange(self.num_nodes)

        # Generate node walks
        walks = []
        for i in range(self.num_walks):
            np.random.shuffle(self.node_indices)
            for j in self.node_indices:
                walk = self._generate_walk(j, self.adjacency_matrix)
                walks.append(walk)

        # Train Word2Vec model on node walks
        model = Word2Vec(
            walks,
            vector_size=self.embedding_size,
            window=self.window_size,
            min_count=0,
            sg=1,
            workers=8,
            epochs=self.num_epochs,
        )

        # Store embedding matrix
        self.embedding_matrix = np.zeros((self.num_nodes, self.embedding_size))
        for i in range(self.num_nodes):
            self.embedding_matrix[i] = model.wv[str(i)]
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        score = []
        for e_pair in X.itertuples(name=None, index=False):
            score.append(
                self.metric_dict[self.metric](
                    self.embedding_matrix[e_pair[0]], self.embedding_matrix[e_pair[1]]
                )
            )
        return np.array(score).reshape(-1, 1)

    def _generate_walk(self, start_node, adjacency_matrix):
        walk = [start_node]
        for i in range(self.walk_length - 1):
            neighbors = np.where(adjacency_matrix[walk[-1]] != 0)[0]
            if len(neighbors) > 0:
                walk.append(np.random.choice(neighbors))
            else:
                break
        return [str(w) for w in walk]


class Node2VecScorer(GraphScorer):
    metric_dict = {"euclidean": euclidean, "cosine": cosine}

    def __init__(
        self,
        input_network,
        num_walks=5,
        walk_length=10,
        embedding_size=50,
        num_negative_samples=4,
        window_size=5,
        num_epochs=10,
        learning_rate=0.01,
        batch_size=1024,
        p=1,
        q=1,
        metric="euclidean",
    ):
        super(Node2VecScorer, self).__init__(input_network)
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.embedding_size = embedding_size
        self.num_negative_samples = num_negative_samples
        self.window_size = window_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.p = p
        self.q = q
        self.metric = metric
        self.adjacency_matrix = nx.to_numpy_array(self.input_network)
        self.history = None
        self.embedding_matrix = None
        self.embedding_model = self.create_model()

    def create_model(self):
        inputs = {
            "target": layers.Input(name="target", shape=(), dtype="int32"),
            "context": layers.Input(name="context", shape=(), dtype="int32"),
        }
        # Initialize item embeddings.
        embed_item = layers.Embedding(
            input_dim=self.input_network.number_of_nodes(),
            output_dim=self.embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
            name="node_embeddings",
        )
        # Lookup embeddings for target.
        target_embeddings = embed_item(inputs["target"])
        # Lookup embeddings for context.
        context_embeddings = embed_item(inputs["context"])
        # Compute dot similarity between target and context embeddings.
        logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
            [target_embeddings, context_embeddings]
        )
        # Create the model.
        model = keras.Model(inputs=inputs, outputs=logits)
        return model

    def next_step(self, previous, current):
        neighbors = list(self.input_network.neighbors(current))
        if len(neighbors) == 0:
            return None
        weights = []
        # Adjust the weights of the edges to the neighbors with respect to p and q.
        for neighbor in neighbors:
            if neighbor == previous:
                # Control the probability to return to the previous node.
                weights.append(1 / self.p)
            elif self.input_network.has_edge(neighbor, previous):
                # The probability of visiting a local node.
                weights.append(1)
            else:
                # Control the probability to move forward.
                weights.append(1 / self.q)

        # Compute the probabilities of visiting each neighbor.
        weight_sum = sum(weights)
        probabilities = [weight / weight_sum for weight in weights]
        # Probabilistically select a neighbor to visit.
        next = np.random.choice(neighbors, size=1, p=probabilities)[0]
        return next

    def generate_walks(self):
        walks = []
        nodes = list(self.input_network.nodes())
        # Perform multiple iterations of the random walk.
        for walk_iteration in range(self.num_walks):
            np.random.shuffle(nodes)

            for node in nodes:
                # Start the walk with a random node from the graph.
                walk = [node]
                # Randomly walk for num_steps.
                while len(walk) < self.walk_length:
                    current = walk[-1]
                    previous = walk[-2] if len(walk) > 1 else None
                    # Compute the next node to visit.
                    next = self.next_step(previous, current)
                    if next is not None:
                        walk.append(next)
                    else:
                        break
                # Replace node ids (movie ids) in the walk with token ids.
                # walk = [vocabulary_lookup[token] for token in walk]
                # Add the walk to the generated sequence.
                walks.append(walk)

        return walks

    def generate_samples(self, walks):
        example_weights = defaultdict(int)
        # Iterate over all sequences (walks).
        for sequence in walks:
            # Generate positive and negative skip-gram pairs for a sequence (walk).
            pairs, labels = keras.preprocessing.sequence.skipgrams(
                sequence,
                vocabulary_size=self.input_network.number_of_nodes(),
                window_size=self.window_size,
                negative_samples=self.num_negative_samples,
            )
            for idx in range(len(pairs)):
                pair = pairs[idx]
                label = labels[idx]
                target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
                if target == context:
                    continue
                entry = (target, context, label)
                example_weights[entry] += 1

        targets, contexts, labels, weights = [], [], [], []
        for entry in example_weights:
            weight = example_weights[entry]
            target, context, label = entry
            targets.append(target)
            contexts.append(context)
            labels.append(label)
            weights.append(weight)

        return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)

    def create_tf_datasets(self, targets, contexts, labels, weights, batch_size):
        inputs = {
            "target": targets,
            "context": contexts,
        }
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
        dataset = dataset.shuffle(buffer_size=batch_size * 2)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset

    def fit(self, X, y=None):
        walks = self.generate_walks()
        targets, contexts, labels, weights = self.generate_samples(walks)
        dataset = self.create_tf_datasets(
            targets=targets,
            contexts=contexts,
            labels=labels,
            weights=weights,
            batch_size=self.batch_size,
        )
        self.embedding_model.compile(
            optimizer=keras.optimizers.Adam(self.learning_rate),
            loss=keras.losses.BinaryCrossentropy(from_logits=True),
        )
        self.history = self.embedding_model.fit(dataset, epochs=self.num_epochs, verbose=0)
        self.embedding_matrix = self.embedding_model.get_layer("node_embeddings").get_weights()[0]
        return self

    def transform(self, X):
        X = self.make_dataset(X)
        score = []
        for e_pair in X.itertuples(name=None, index=False):
            score.append(
                self.metric_dict[self.metric](
                    self.embedding_matrix[e_pair[0]], self.embedding_matrix[e_pair[1]]
                )
            )
        return np.array(score).reshape(-1, 1)
