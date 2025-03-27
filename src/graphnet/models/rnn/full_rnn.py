import torch
import torch.nn as nn
from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data
from typing import List, Optional
from graphnet.models.components.embedding import SinusoidalPosEmb


class Full_RNN(GNN):
    """Full RNN model capable of producing final predictions."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        hidden_size: int,
        num_layers: int,
        time_series_columns: List[int],
        nb_neighbours: int = 8,
        features_subset: Optional[List[int]] = None,
        dropout: float = 0.5,
        embedding_dim: int = 0,
        output_size: int = 1,  # Number of output features (e.g., regression or classification)
    ) -> None:
        """Construct `Full_RNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            time_series_columns: The indices of the input data that should be treated as time series data.
            nb_neighbours: Number of neighbours to use when reconstructing the graph representation.
            features_subset: The subset of latent features on each node that are used as metric dimensions.
            dropout: Dropout fraction to use in the RNN.
            embedding_dim: Dimension of the embedding.
            output_size: Number of features in the final output.
        """
        super().__init__(nb_inputs, hidden_size + 5)

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._time_series_columns = time_series_columns
        self._nb_neighbors = nb_neighbours
        self._features_subset = features_subset
        self._embedding_dim = embedding_dim
        self._nb_inputs = nb_inputs

        # RNN layer
        self.rnn = nn.LSTM(
            input_size=nb_inputs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the Full RNN model.

        Args:
            data: Input graph data.

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        # Extract time-series data from the input
        x = data.x  # Node features

        # Apply RNN
        rnn_out, _ = self.rnn(x)  # rnn_out: (batch_size, sequence_length, hidden_size)

        # Use the last hidden state for each sequence
        last_hidden_state = rnn_out[:, -1, :]  # (batch_size, hidden_size)

        # Apply the fully connected layer to produce final predictions
        output = self.fc(last_hidden_state)  # (batch_size, output_size)

        return output