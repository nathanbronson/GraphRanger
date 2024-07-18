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
import numpy as np

from graph import Graph, obgnmag
from model import GraphModel

BATCH_SIZE = 8
PATH_LENGTH = 8

if __name__ == "__main__":
    g = Graph.load("./training_graph.grph")#obgnmag()
    g.prepare_for_training(save=False)
    ds = g.as_mapped_dataset(batch_size=BATCH_SIZE, path_length=PATH_LENGTH)
    example = None
    for i in ds.take(1):
        example = i
    model = GraphModel(g.edge_vocab_size, g.node_vocab_size, classes=np.unique(g.node_labels.numpy()).shape[0])
    model.test_step(example)
    model.built = True
    model.compile()
    print(model.summary())
    model.fit(ds, steps_per_epoch=512, epochs=100)