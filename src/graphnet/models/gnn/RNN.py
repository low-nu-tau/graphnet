import torch
from torch import nn
from torch_geometric.data import Data
from typing import List, Optional
from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config


class String_RNN(GNN):
    """RNN model for aggregated features of the string with the highest charge."""

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        hidden_size: int,
        num_layers: int,
        time_series_columns: List[int],
        dropout: float = 0.5,
        embedding_dim: int = 0,
    ) -> None:
        """Initialize the String_RNN model.

        Args:
            nb_inputs: Number of input features.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            time_series_columns: Indices of the input data treated as time series.
            dropout: Dropout fraction to use in the RNN. Defaults to 0.5.
            embedding_dim: Embedding dimension of the RNN. Defaults to 0.
        """
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._time_series_columns = time_series_columns
        self._embedding_dim = embedding_dim
        self._nb_inputs = nb_inputs

        super().__init__(nb_inputs, hidden_size)

        if self._embedding_dim != 0:
            self._nb_inputs = self._embedding_dim * nb_inputs

        self._rnn = nn.GRU(
            input_size=self._nb_inputs,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=dropout,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass for the String_RNN model.

        Args:
            data: Input data object.

        Returns:
            torch.Tensor: Final string score.
        """
        x = data.x
        batch = data.batch

        # Aggregate features for the string with the highest charge
        string_ids = data.string_ids
        charges = x[:, self._time_series_columns[0]]
        string_charges = torch.zeros_like(charges).scatter_add_(
            0, string_ids, charges
        )
        max_string_id = string_charges.argmax()

        # Select features for the string with the highest charge
        string_mask = string_ids == max_string_id
        string_features = x[string_mask]

        # Optional embedding
        if self._embedding_dim != 0:
            string_features = string_features * 4096
            string_features = string_features.view(
                string_features.size(0), -1, self._embedding_dim
            )

        # Apply RNN
        rnn_out, _ = self._rnn(string_features.unsqueeze(0))
        string_score = rnn_out[:, -1, :]  # Take the last hidden state

        return string_score