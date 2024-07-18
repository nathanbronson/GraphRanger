<p align="center"><img src="https://github.com/nathanbronson/GraphRanger/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# GraphRanger
neural networks for heterogeneous data

## About
*GraphRanger is a work-in-progress. Some features are still unstable or need to be implemented and optimized. Updates to this repository will come periodically as these changes are made.*

GraphRanger is a strategy for embedding highly hetergeneous data for use in conventional neural networks. It uses a navigator (the ranger) to compile strings of adjacent nodes and edges then uses discrete embeddings to vectorize each edge-node combination.

The ranger is trained to navigate the graph such that the string it compiles minimizes the greater model loss. It is trained as a reinforcement learning agent with a reward network to predict the marginal loss of a movement in the graph then move such that it minimizes this marginal loss. It chooses its moves using a Monte Carlo tree search.

GraphRanger is meant to be integrated with conventional neural networks. The example included in this repo uses the ranger to embed information as input to an otherwise unaltered transformer model to make predictions on the OGBN-MAG dataset.

This codebase includes the code for the GraphRanger embedding model integrated with a test example of the model in use (`model.py`). It also includes a class to handle graphs. This class ingests and preprocesses a graph for efficient ingestion by the ranger (`graph.py`). These utilities prioritize quick retrieval of nodes and edges adjacent to a given node. The class also allows serialized storage of the graphs in this processed form on disk.

## Usage
To run the example, execute `test.py`. This example is inference on the OGBN-MAG dataset.

## License
See `LICENSE`.