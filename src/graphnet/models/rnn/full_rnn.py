import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import knn_graph
from typing import List, Optional
from graphnet.models.gnn.gnn import GNN
from graphnet.utilities.config import save_model_config
from graphnet.models.components.embedding import SinusoidalPosEmb


class Full_RNN(GNN):
    """Standalone RNN model for time-series data."""

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
        output_size: int = 1,
        embedding_dim: int = 0,
    ) -> None:
        """Construct `Full_RNN`.

        Args:
            nb_inputs: Number of features in the input data.
            hidden_size: Number of features for the RNN output and hidden layers.
            num_layers: Number of layers in the RNN.
            time_series_columns: Indices of the input data treated as time-series data.
            nb_neighbours: Number of neighbours for graph reconstruction. Defaults to 8.
            features_subset: Subset of latent features used for k-NN clustering. Defaults to [0, 1, 2, 3].
            dropout: Dropout fraction for the RNN. Defaults to 0.5.
            output_size: Number of output features (e.g., regression targets or scores). Defaults to 1.
            embedding_dim: Dimension of the positional embedding. Defaults to 0.
        """
        super().__init__(nb_inputs, hidden_size)

        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._time_series_columns = time_series_columns
        self._nb_neighbors = nb_neighbours
        self._features_subset = features_subset or [0, 1, 2, 3]
        self._embedding_dim = embedding_dim
        self._nb_inputs = nb_inputs
        self._output_size = output_size

        if self._embedding_dim != 0:
            self._nb_inputs = self._embedding_dim * nb_inputs

        # RNN layer
        self._rnn = torch.nn.GRU(
            num_layers=self._num_layers,
            input_size=self._nb_inputs,
            hidden_size=self._hidden_size,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected output layer
        self._fc = torch.nn.Linear(self._hidden_size, self._output_size)

        # Optional sinusoidal positional embedding
        self._emb = SinusoidalPosEmb(dim=self._embedding_dim)

    def clean_up_data_object(self, data: Data) -> Data:
        """Update the feature names of the data object.

        Args:
            data: The input data object.
        """
        # old features removing the new_node column
        old_features = data.features[0][:-1]
        new_features = old_features + [
            "rnn_out_" + str(i) for i in range(self._hidden_size)
        ]
        data.features = [new_features] * len(data.features)
        for i, name in enumerate(old_features):
            data[name] = data.x[i]
        return data
    
    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        time_series = x[:, self._time_series_columns]

        # Apply positional embedding if specified
        if self._embedding_dim != 0:
            time_series = self._emb(time_series * 4096).reshape(
            (
                time_series.shape[0],
                self._embedding_dim * time_series.shape[-1],
            )
            )

        # Run the RNN
        rnn_out, _ = self._rnn(time_series.unsqueeze(1))  # Add batch dimension

        # Take the last hidden state
        rnn_out = rnn_out[:, -1, :]

        # Fully connected layer to produce the final output
        output = self._fc(rnn_out)

        return output