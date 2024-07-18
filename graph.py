"""
GraphRanger: neural networks for heterogenous data
Copyright (C) 2024  Nathan Bronson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pickle

from tqdm import tqdm

import tensorflow as tf
import tensorflow_gnn as tfgnn

from utils import random_pair, random_items
from disk_utils import write_tensor_list, read_tensor_list

NECESSARY_VARS = [
    "adjacent_node_ids",
    "adjacent_edge_ids",
    "adjacent_edges_directions",
    "adjacent_directioned_edge_ids",
    "node_id_vocab",
    "edge_id_vocab",
    "nodes",
    "edges",
    "adjacent_nodes",
    "adjacent_edges",
    "edge_sources",
    "edge_targets"
]
TYPE_DICT = {
    "num_nodes": tf.int64,
    "node_ids": tf.string,
    "node_labels": tf.float32,
    "num_edges": tf.int64,
    "edge_ids": tf.string,
    "adjacent_node_ids": tf.string,
    "adjacent_edge_ids": tf.string,
    "adjacent_edges_directions": tf.int32,
    "adjacent_directioned_edge_ids": tf.int32,
    "node_id_vocab": None,#tf.io.serialize_tensor(self.node_id_vocab),
    "edge_id_vocab": None,#tf.io.serialize_tensor(self.edge_id_vocab),
    "nodes": tf.int32,
    "edges": tf.int32,
    "adjacent_nodes": tf.int32,
    "adjacent_edges": tf.int32,
    "edge_sources": tf.int32,
    "edge_targets": tf.int32
}
EXTRA_ATTRS = [
    "node_ids",
    "edge_ids",
    "adjacent_node_ids",
    "adjacent_edge_ids",
    "adjacent_edges_directions",
    "edge_source",
    "edge_targets"
]

class Graph:
    def __init__(self, *, base_graph: tfgnn.GraphTensor=None, **kwargs):
        if base_graph is not None:
            #assert base_graph is not None, f"`base_graph` cannot be None if one of the initializer variables is None"
            for k, v in kwargs.items():
                setattr(self, k, v)
            self.build_from_base_graph(base_graph)
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    def build_from_base_graph(self, base_graph: tfgnn.GraphTensor):
        print("saving base adjacency")
        self.edge_sources = base_graph.edge_sets["edges"].adjacency.source
        self.edge_targets = base_graph.edge_sets["edges"].adjacency.target

        print("building node adjacency")
        self.adjacent_node_ids = self._adjacent_nodes()
        print("building edge adjacency")
        self.adjacent_edge_ids, self.adjacent_edges_directions = self._adjacent_edges()
        print("directioning edges")
        self.adjacent_directioned_edge_ids = tf.ragged.constant([[i + g for i, g in zip(_aie, _d)] for _aie, _d in tqdm(list(zip(self.adjacent_edge_ids.numpy().tolist(), tf.as_string(self.adjacent_edges_directions).numpy().tolist())))])
        
        print("building node vocab")
        self.node_id_vocab, self.node_vocab_size = Graph.make_vocab(self.node_ids)
        print("building edge vocab")
        self.edge_id_vocab, self.edge_vocab_size = Graph.make_vocab(tf.reshape(self.adjacent_directioned_edge_ids, (-1,)))
        
        print("indexing nodes")
        self.nodes = self.node_id_vocab.lookup(self.node_ids)
        assert tf.reduce_all(self.nodes[:-1] == tf.range(tf.shape(self.nodes)[0] - 1) + 1), ("`nodes` is ordered irregularly", self.nodes)
        #print("indexing edges")
        #self.edges = self.edge_id_vocab.lookup(self.edge_ids)
        #assert tf.reduce_all(self.edges[:-1] == tf.range(tf.shape(self.edges)[0] - 1) + 1), ("`edges` is ordered irregularly", self.edges)
        
        print("indexing adjacency")
        self.adjacent_nodes = self.node_id_vocab.lookup(self.adjacent_node_ids)
        self.adjacent_edges = self.edge_id_vocab.lookup(self.adjacent_directioned_edge_ids)
        self.adjacent_nodes = tf.concat([self.adjacent_nodes, tf.convert_to_tensor([[0]])], axis=0)
        self.adjacent_edges = tf.concat([self.adjacent_edges, tf.convert_to_tensor([[0]])], axis=0)
        
        print("completed graph build")
    
    def _prune(self):
        for i in EXTRA_ATTRS:
            if hasattr(self, i):
                delattr(self, i)
    
    def prepare_for_training(self, save=True, path="graph.pkl"):
        if save:
            self.to_pkl(path)
        self._prune()
    
    @classmethod
    def from_tfgnn_path(cls, graph_path, schema_path, is_homogenized=False, id_key="#id"):
        g = read_graph(graph_path, schema_path)
        if not is_homogenized:
            return cls.from_generic_tfgnn_graph(g, id_key=id_key)
        return cls.from_blueprint(g)
    
    @classmethod
    def from_generic_tfgnn_graph(cls, graph, id_key="#id"):
        using_label = "label" in [n for k in graph.node_sets.keys() for n in graph.node_sets[k].features.keys()]
        node_reindexer = {}
        node_top_level_counter = 0
        labels = []
        print("reindexing nodes")
        ns_keys = list(graph.node_sets.keys())
        for k in tqdm(ns_keys):
            node_reindexer[k] = 0
            feat_names = list(graph.node_sets[k].features.keys())
            features = graph.node_sets[k].features
            node_reindexer[k] = node_top_level_counter
            num_in = features[id_key].shape[0]
            node_top_level_counter += num_in
            if using_label:
                if "label" in feat_names:
                    labels.extend(features["label"].numpy())
                else:
                    labels.extend([-1] * num_in)
        edge_ids = []
        edge_top_level_counter = 0
        print("reindexing edges")
        es_keys = list(graph.edge_sets.keys())
        for k in tqdm(es_keys):
            n = graph.edge_sets[k].adjacency.source.shape[0]
            add = [k] * n
            edge_top_level_counter += n
            edge_ids.extend(add)
        print("making new node set")
        new_node_set = tfgnn.NodeSet.from_fields(
            sizes=tf.convert_to_tensor([node_top_level_counter]),
            features={
                "id": tf.range(node_top_level_counter)
            } | ({} if not using_label else {
                "label": tf.convert_to_tensor(labels)
            })
        )
        print("making new edge set")
        print("reindexing sources")
        all_sources = tf.concat([node_reindexer[graph.edge_sets[k].adjacency.source_name] + graph.edge_sets[k].adjacency.source for k in tqdm(es_keys)], axis=0)
        print("reindexing targets")
        all_targets = tf.concat([node_reindexer[graph.edge_sets[k].adjacency.target_name] + graph.edge_sets[k].adjacency.target for k in tqdm(es_keys)], axis=0)
        new_edge_set =  tfgnn.EdgeSet.from_fields(
            sizes=tf.convert_to_tensor([edge_top_level_counter]),
            features={
                "id": tf.convert_to_tensor(edge_ids)
            },
            adjacency=tfgnn.Adjacency.from_indices(
                source=("nodes", all_sources),
                target=("nodes", all_targets)
            )
        )
        print("loading graph")
        return cls.from_blueprint(tfgnn.GraphTensor.from_pieces(
            node_sets={"nodes": new_node_set},
            edge_sets={"edges": new_edge_set}
        ))
    
    @classmethod
    def from_blueprint(cls, base_graph: tfgnn.GraphTensor):
        print("validating and loading graph")
        assert list(base_graph.node_sets.keys()) == ["nodes"], list(base_graph.node_sets.keys())
        assert all([k in ["id", "label"] for k in list(base_graph.node_sets["nodes"].features.keys())]), list(base_graph.node_sets["nodes"].features.keys())
        assert list(base_graph.edge_sets.keys()) == ["edges"], list(base_graph.edge_sets.keys())
        assert list(base_graph.edge_sets["edges"].features.keys()) == ["id"], list(base_graph.edge_sets["edges"].features.keys())
        num_nodes = base_graph.node_sets["nodes"].features["id"].shape[0]
        node_ids = tf.as_string(base_graph.node_sets["nodes"].features["id"])
        assert all(["nan" != i for i in node_ids.numpy()]), "'nan' is a reserved id and cannot be in `node_ids`"
        node_labels = base_graph.node_sets["nodes"].features["label"] if "label" in base_graph.node_sets["nodes"].features else None
        if node_labels is not None:
            node_labels = tf.concat([node_labels, tf.convert_to_tensor([-1])], axis=0)
        num_edges = base_graph.edge_sets["edges"].adjacency.source.shape[0]
        edge_ids = tf.as_string(base_graph.edge_sets["edges"].features["id"])
        assert all(["nan" != i for i in edge_ids.numpy()]), "'nan' is a reserved id and cannot be in `edge_ids`"
        assert all(["init" != i for i in edge_ids.numpy()]), "'init' is a reserved id and cannot be in `edge_ids`"
        print("graph validated and loaded")
        print("creating object")
        return cls(base_graph=base_graph, num_nodes=num_nodes, node_ids=node_ids, node_labels=node_labels, num_edges=num_edges, edge_ids=edge_ids)

    def to_blueprint(self, graph_path, schema_path):
        return write_graph(tfgnn.GraphTensor.from_pieces(
            node_sets={
                "nodes": tfgnn.NodeSet.from_fields(
                    sizes=self.num_nodes,
                    features={
                        "id": self.node_ids
                    } | ({} if self.node_labels is not None else {
                        "label": self.node_labels
                    })
                )
            },
            edge_sets={
                "edges": tfgnn.EdgeSet.from_fields(
                    sizes=self.num_edges,
                    features={
                        "id": self.edge_ids
                    },
                    adjacency=tfgnn.Adjacency.from_indices(
                        source=self.edge_sources,
                        target=self.edge_targets
                    )
                )
            }
        ), graph_path, schema_path)

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as doc:
            return cls(**{k: tf.io.parse_tensor(v, TYPE_DICT[k]) for k, v in pickle.load(doc).items()})

    def to_file(self, path):
        with open(path, "wb") as doc:
            pickle.dump(self.to_dict(), doc)
    
    @classmethod
    def load(cls, path):
        ek, ev, nk, nv, nodes, num_nodes, num_edges, node_labels, anv, anl, aev, ael = read_tensor_list(path, 12)
        return cls(
            edge_id_vocab=Graph.vocab_from_kv(ek, ev),
            node_id_vocab=Graph.vocab_from_kv(nk, nv),
            nodes=nodes,
            num_nodes=num_nodes.numpy(),
            num_edges=num_edges.numpy(),
            node_labels=node_labels,
            adjacent_edges=tf.RaggedTensor.from_row_lengths(aev, row_lengths=ael),
            adjacent_nodes=tf.RaggedTensor.from_row_lengths(anv, row_lengths=anl),
            node_vocab_size=tf.unique(nv)[0].shape[0],
            edge_vocab_size=tf.unique(ev)[0].shape[0]
        )

    def save(self, path):
        return write_tensor_list([
            self.edge_id_vocab._initializer._keys,
            self.edge_id_vocab._initializer._values,
            self.node_id_vocab._initializer._keys,
            self.node_id_vocab._initializer._values,
            self.nodes,
            tf.convert_to_tensor(self.num_nodes),
            tf.convert_to_tensor(self.num_edges),
            self.node_labels,
            self.adjacent_nodes.values,
            self.adjacent_nodes.row_lengths(),
            self.adjacent_edges.values,
            self.adjacent_edges.row_lengths()
        ], path)
    
    @classmethod
    def from_pkl(cls, path):
        with open(path, "rb") as doc:
            obj = pickle.load(doc)
        setattr(obj, "edge_id_vocab", Graph.vocab_from_kv(obj.edge_keys, obj.edge_values))
        setattr(obj, "node_id_vocab", Graph.vocab_from_kv(obj.node_keys, obj.node_values))
        return obj
    
    def to_pkl(self, path):
        self.edge_keys = self.edge_id_vocab._initializer._keys
        self.edge_values = self.edge_id_vocab._initializer._values
        del self.edge_id_vocab
        self.node_keys = self.node_id_vocab._initializer._keys
        self.node_values = self.node_id_vocab._initializer._values
        del self.node_id_vocab
        with open(path, "wb") as doc:
            res = pickle.dump(self, doc)
        self.edge_id_vocab = Graph.vocab_from_kv(self.edge_keys, self.edge_values)
        self.node_id_vocab = Graph.vocab_from_kv(self.node_keys, self.node_values)
        return res

    def to_dict(self):
        return {
            "num_nodes": tf.io.serialize_tensor(self.num_nodes),
            "node_ids": tf.io.serialize_tensor(self.node_ids),
            "node_labels": tf.io.serialize_tensor(self.node_labels),
            "num_edges": tf.io.serialize_tensor(self.num_edges),
            "edge_ids": tf.io.serialize_tensor(self.edge_ids),
            "adjacent_node_ids": tf.io.serialize_tensor(self.adjacent_node_ids),
            "adjacent_edge_ids": tf.io.serialize_tensor(self.adjacent_edge_ids),
            "adjacent_edges_directions": tf.io.serialize_tensor(self.adjacent_edges_directions),
            "adjacent_directioned_edge_ids": tf.io.serialize_tensor(self.adjacent_directioned_edge_ids),
            "node_id_vocab": tf.io.serialize_tensor(self.node_id_vocab),
            "edge_id_vocab": tf.io.serialize_tensor(self.edge_id_vocab),
            "nodes": tf.io.serialize_tensor(self.nodes),
            "edges": tf.io.serialize_tensor(self.edges),
            "adjacent_nodes": tf.io.serialize_tensor(self.adjacent_nodes),
            "adjacent_edges": tf.io.serialize_tensor(self.adjacent_edges),
            "edge_sources": tf.io.serialize_tensor(self.edge_sources),
            "edge_targets": tf.io.serialize_tensor(self.edge_targets)
        }
    
    def _adjacent_nodes(self):#iterate through adjacency and append instead of nodes like this
        adjs = [[] for _ in range(self.num_nodes)]
        pair = tf.concat([self.edge_sources, self.edge_targets], axis=0).numpy()
        nids = self.node_ids.numpy()
        for n, i in tqdm(list(enumerate(tf.concat([self.edge_targets, self.edge_sources], axis=0).numpy()))):
            adjs[pair[n]].append(nids[i])
        return tf.ragged.constant(adjs)
        #_adj = [self._do_adj_n(tf.constant(i, dtype=tf.int64)) for i in tqdm(range(self.num_nodes))]
        #return tf.concat(_adj, axis=0)
    
    @tf.function
    def _do_adj_n(self, idx):
        return tf.expand_dims(tf.concat([
            tf.gather(self.node_ids, self.edge_sources[self.edge_targets == idx]),
            tf.gather(self.node_ids, self.edge_targets[self.edge_sources == idx])
        ], axis=0), 0)

    def _adjacent_edges(self):
        adjs = [[] for _ in range(self.num_nodes)]
        et = self.edge_targets.numpy()
        es = self.edge_sources.numpy()
        dirs =  [[] for _ in range(self.num_nodes)]
        for n, i in tqdm(list(enumerate(self.edge_ids.numpy()))):
            etn = et[n]
            adjs[etn].append(i)
            dirs[etn].append(-1)
        for n, i in tqdm(list(enumerate(self.edge_ids.numpy()))):
            esn = es[n]
            adjs[esn].append(i)
            dirs[esn].append(1)
        return tf.ragged.constant(adjs), tf.ragged.constant(dirs)
        #res = [self._do_adj_e(tf.constant(i, dtype=tf.int64)) for i in tqdm(range(self.num_nodes))]
        #return tf.concat([i[0] for i in res], axis=0), tf.concat([i[1] for i in res], axis=0)
    
    @tf.function
    def _do_adj_e(self, idx):
        tar = self.edge_ids[self.edge_targets == idx]
        src = self.edge_ids[self.edge_sources == idx]
        _adj = tf.expand_dims(tf.concat([
            src,
            tar
        ], axis=0), 0)
        _dir = tf.expand_dims(tf.concat([
            -tf.ones_like(tar),
            tf.ones_like(src)
        ], axis=0), 0)
        return _adj, _dir
    
    @tf.function
    def get_node_info_from_id(self, node_ids):
        node_idx = self.node_id_vocab.lookup(node_ids) - 1
        #assert tf.reduce_all(node_ids == tf.gather(self.node_ids, node_idx))
        adj_edges = self.edge_id_vocab.lookup(tf.gather(self.adjacent_directioned_edge_ids, node_idx))
        adj_nodes = self.node_id_vocab.lookup(tf.gather(self.adjacent_node_ids, node_idx))
        return node_idx, adj_edges, adj_nodes
    
    @tf.function
    def get_node_info(self, nodes):
        node_idx = nodes - 1
        adj_edges = tf.gather(self.adjacent_edges, node_idx, axis=0)
        adj_nodes = tf.gather(self.adjacent_nodes, node_idx, axis=0)
        return adj_edges, adj_nodes
    
    @tf.function
    def random_neighbor(self, nodes):
        adj_edges, adj_nodes = self.get_node_info(nodes)
        adj_info = tf.concat([tf.expand_dims(adj_edges, -1), tf.expand_dims(adj_nodes, -1)], axis=-1)
        return random_pair(adj_info)

    @tf.function
    def sample_paths(self, size, sequence_length):
        init_token = self.edge_id_vocab.lookup(tf.convert_to_tensor("init"))
        paths = tf.concat([tf.ones((size, 1), dtype=tf.int32) * init_token, tf.expand_dims(random_items(self.start_nodes, size), -1)], axis=-1)
        paths = tf.expand_dims(paths, -2)
        for _ in range(sequence_length - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(paths, tf.TensorShape([None, None, 2]))]
            )
            chosen_neighbor = self.random_neighbor(paths[:, -1, 1])
            paths = tf.concat([paths, tf.expand_dims(chosen_neighbor, 1)], axis=-2)
        return paths
    
    @tf.function
    def get_labels(self, nodes):
        assert self.node_labels is not None
        return tf.gather(self.node_labels, nodes - 1)
    
    @tf.function
    def get_batch(self, size, length):
        paths = self.sample_paths(size, length)
        x = tf.RaggedTensor.from_tensor(paths[:, :0 + 1])
        for i in range(1, length):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(x, tf.TensorShape([None, None, 2]))]
            )
            x = tf.concat([x, tf.RaggedTensor.from_tensor(paths[:, :i + 1])], axis=0)
        marginal_node = tf.reshape(tf.gather_nd(x, tf.concat([tf.expand_dims(tf.range(tf.shape(x)[0], dtype=tf.int64), -1), tf.expand_dims(x.row_lengths(1) - 1, -1)], axis=-1)), (-1, 2))
        s0 = tf.shape(paths)[0]
        current_mask = tf.reshape(tf.map_fn(lambda i: tf.range(s0) + (i * s0), tf.range(length - 1)), (-1,))
        forward_looking_mask = tf.reshape(tf.map_fn(lambda i: tf.range(s0) + (i * s0), tf.range(1, length)), (-1,))
        idx = tf.concat([tf.expand_dims(tf.range(tf.shape(x)[0]), -1), tf.repeat(tf.expand_dims(tf.constant([0, 1]), 0), tf.shape(x)[0], axis=0)], axis=-1)
        y = self.get_labels(tf.gather_nd(x, idx))
        return (x.to_tensor(), tf.gather(marginal_node, forward_looking_mask, axis=0), current_mask, forward_looking_mask), y
    
    @tf.function
    def _get_batch(self, inputs):
        size = inputs[0]
        length = inputs[1]
        return self.get_batch(size, length)

    @staticmethod
    def vocab_from_kv(k, v):
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                k,
                v
            ), default_value=-1
        )
    
    @staticmethod
    def make_vocab(ids):
        u, _ = tf.unique(ids)
        vocab = {i: n + 1 for n, i in tqdm(list(enumerate(u.numpy())))} | {"init": tf.shape(u)[0] + 1} | {"nan": 0}
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.convert_to_tensor(list(vocab.keys())),
                tf.convert_to_tensor(list(vocab.values()))
            ), default_value=-1
        ), len(vocab)
    
    def as_generator_dataset(self, batch_size=16, path_length=16):
        bs = tf.constant(batch_size)
        pl = tf.constant(path_length)
        def gen():
            while True:
                yield self.get_batch(bs, pl)
        return tf.data.Dataset.from_generator(
            gen,
            output_types=((tf.int32, tf.int32, tf.int32, tf.int32), tf.int32),
            output_shapes=(((None, path_length, 2), (None, 1, 2), (None,), (None,)), (None,))
        )
    
    def as_mapped_dataset(self, batch_size=16, path_length=16, exclude_label=-1):
        if exclude_label is not None and self.node_labels is not None:
            self.start_nodes = self.nodes[self.node_labels[:-1] != exclude_label]
        else:
            self.start_nodes = self.nodes
        return tf.data.Dataset.from_tensor_slices(
            tf.constant([[batch_size, path_length]])
        ).repeat().map(self._get_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)

def test_graph(size=3):
    return tfgnn.GraphTensor.from_pieces(
        node_sets={
            "nodes": tfgnn.NodeSet.from_fields(
                sizes=tf.constant([size]),
                features={
                    "id": tf.ones((size,))
                }
            )
        },
        edge_sets={
            "edges": tfgnn.EdgeSet.from_fields(
                sizes=tf.constant([3]),
                features={
                    "id": tf.ones((size,))
                },
                adjacency=tfgnn.Adjacency.from_indices(
                    source=("nodes", tf.cast(tf.random.uniform((size,), 0, size) * size, tf.int32)),
                    target=("nodes", tf.cast(tf.random.uniform((size,), 0, size) * size, tf.int32))
                )
            )
        }
    )

def write_graph(graph, graph_path, schema_path):
    tfgnn.write_schema(tfgnn.create_schema_pb_from_graph_spec(graph), schema_path)
    with tf.io.TFRecordWriter(graph_path) as writer:
        example = tfgnn.write_example(graph)
        writer.write(example.SerializeToString())

def read_graph(graph_path, schema_path):
    graph_spec = tfgnn.create_graph_spec_from_schema_pb(tfgnn.read_schema(schema_path))
    dataset = tf.data.TFRecordDataset(filenames=[graph_path])
    dataset = dataset.map(lambda s: tfgnn.parse_single_example(graph_spec, s))
    for graph in dataset.take(1):
        return graph

def obgnmag():
    GRAPH_TENSOR_FILE = 'gs://download.tensorflow.org/data/ogbn-mag/sampled/v2/graph_tensor.example.pb'
    SCHEMA_FILE = 'gs://download.tensorflow.org/data/ogbn-mag/sampled/v2/graph_schema.pbtxt'

    graph_schema = tfgnn.read_schema(SCHEMA_FILE)
    serialized_ogbn_mag_graph_tensor_string = tf.io.read_file(GRAPH_TENSOR_FILE)

    g = tfgnn.parse_single_example(
        tfgnn.create_graph_spec_from_schema_pb(graph_schema, indices_dtype=tf.int64),
        serialized_ogbn_mag_graph_tensor_string)
    
    print("loading graph")
    return Graph.from_generic_tfgnn_graph(g)

if __name__ == "__main__":
    GRAPH_TENSOR_FILE = 'gs://download.tensorflow.org/data/ogbn-mag/sampled/v2/graph_tensor.example.pb'
    SCHEMA_FILE = 'gs://download.tensorflow.org/data/ogbn-mag/sampled/v2/graph_schema.pbtxt'

    graph_schema = tfgnn.read_schema(SCHEMA_FILE)
    serialized_ogbn_mag_graph_tensor_string = tf.io.read_file(GRAPH_TENSOR_FILE)

    g = tfgnn.parse_single_example(
        tfgnn.create_graph_spec_from_schema_pb(graph_schema, indices_dtype=tf.int64),
        serialized_ogbn_mag_graph_tensor_string)
    
    print("loading graph")
    my_graph = Graph.from_generic_tfgnn_graph(g)
    #print("saving graph")
    #my_graph.to_pkl("./graph.pkl")
    print("saving training graph")
    my_graph.prepare_for_training(save=False)
    my_graph.save("./training_graph.grph")
    #my_graph.to_pkl("./training_graph.pkl")

    print("reloading")
    reloaded = Graph.load("./training_graph.grph")