"""RNN model implementation."""

from typing import List, Optional

import torch
from graphnet.models.gnn.gnn import GNN
from graphnet.models.rnn.node_rnn import Node_RNN

from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class RNN(GNN):
    """The RNN model class.

    Combines the Node_RNN model, intended for data with large
    amount of DOM activations per event. This model works only with non-
    standard dataset specific to the Node_RNN model see Node_RNN for more
    details.
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        time_series_columns: List[int],
        *,
        nb_neighbours: int = 8,
        rnn_layers: int = 2,
        rnn_hidden_size: int = 64,
        rnn_dropout: float = 0.5,
        features_subset: Optional[List[int]] = None,
        embedding_dim: Optional[int] = None,
    ):
        """Initialize the RNN model.

        Args:
            nb_inputs (int): Number of input features.
            time_series_columns (List[int]): The indices of the input data that
                should be treated as time series data.
                The first index should be the charge column.
            nb_neighbours (int, optional): Number of neighbours to consider.
                Defaults to 8.
            rnn_layers (int, optional): Number of RNN layers.
                Defaults to 2.
            rnn_hidden_size (int, optional): Size of the hidden state of the
                RNN. Also determines the size of the output of the RNN.
                Defaults to 64.
            rnn_dropout (float, optional): Dropout to use in the RNN.
                Defaults to 0.5.
            features_subset (List[int], optional): The subset of latent
                features on each node that are used as metric dimensions when
                performing the k-nearest neighbours clustering.
                Defaults to None.
            embedding_dim (int, optional): Embedding dimension of the RNN.
                Defaults to None.
        """
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = nb_inputs
        self._rnn_layers = rnn_layers
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_dropout = rnn_dropout
        self._embedding_dim = embedding_dim

        self._features_subset = features_subset

        super().__init__(nb_inputs, self._rnn_hidden_size)

        self._rnn = Node_RNN(
            nb_inputs=2,
            hidden_size=self._rnn_hidden_size,
            num_layers=self._rnn_layers,
            time_series_columns=time_series_columns,
            nb_neighbours=self._nb_neighbours,
            features_subset=self._features_subset,
            dropout=self._rnn_dropout,
            embedding_dim=self._embedding_dim,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass of the RNN model."""
        readout = self._rnn(data)
        return readout
