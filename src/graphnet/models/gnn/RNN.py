"""RNN_TITO model implementation."""

from typing import List, Optional

import torch
from graphnet.models.gnn.gnn import GNN
from graphnet.models.rnn.full_rnn import Full_RNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class RNN(GNN):
    """The RNN_TITO model class.

    Modified to only perform the RNN layer and return the RNN output.
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
        """Initialize the RNN_TITO model.

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
                Defaults to [0,1,2,3].
            embedding_dim (int, optional): Embedding dimension of the RNN.
                Defaults to None (no embedding).
        """
        self._nb_neighbours = nb_neighbours
        self._nb_inputs = nb_inputs
        self._rnn_layers = rnn_layers
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_dropout = rnn_dropout
        self._embedding_dim = embedding_dim

        self._features_subset = features_subset

        super().__init__(nb_inputs, self._rnn_hidden_size)

        # Initialize the RNN layer
        self._rnn = Full_RNN(
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
        """Apply learnable forward pass of the RNN model.

        Args:
            data: Input graph data.

        Returns:
            RNN output tensor.
        """
        # Debug input data
        # print(f"[DEBUG] In RNN Input data.x shape: {data.x.shape}")
        # print(f"[DEBUG] In RNN Input data.edge_index shape: {data.edge_index.shape}")

        # Pass the data through the RNN layer
        rnn_out = self._rnn(data)

        # Debug output data
        # print(f"[DEBUG] RNN output shape: {rnn_out.shape}")

        # Return the RNN output
        return rnn_out