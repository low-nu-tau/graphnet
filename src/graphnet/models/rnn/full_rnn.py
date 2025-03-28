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

        # RNN layer
        self._rnn = torch.nn.GRU(
            input_size=len(time_series_columns),
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected output layer
        self._fc = torch.nn.Linear(self._hidden_size, self._output_size)

        # Optional sinusoidal positional embedding
        self._emb = SinusoidalPosEmb(dim=self._embedding_dim) if self._embedding_dim > 0 else None
   
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass of the Full RNN model.

        Args:
            data: Input graph data.

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        x = data.x  # Node features
        batch = data.batch  # Batch indices

        # Extract time-series data
        time_series = x[:, self._time_series_columns]

        # Apply positional embedding if specified
        if self._embedding_dim > 0:
            time_series = self._emb(time_series * 4096).reshape(
                time_series.shape[0], -1
            )

        # Create the DOM + batch unique splitter from the new_node_col
        splitter = torch.cat([torch.tensor([0]), x[:, -1].argwhere().flatten().cpu()])
        print(f"Splitter values: {splitter}")

        # Split the time-series data
        time_series = time_series.tensor_split(splitter)
        print(f"Lengths of time_series splits before filtering: {[len(ts) for ts in time_series]}")

        # Filter out empty splits
        time_series = [ts for ts in time_series if len(ts) > 0]
        print(f"Lengths of time_series splits after filtering: {[len(ts) for ts in time_series]}")

        # Apply RNN per DOM irrespective of batch and return the final state
        time_series = torch.nn.utils.rnn.pack_sequence(
            time_series, enforce_sorted=False
        )
        rnn_out = self._rnn(time_series)[-1][0]  # Extract the final hidden state

        # Correct the batch tensor
        valid_indices = x[:, -1].bool()
        print(f"Number of True values in x[:, -1]: {valid_indices.sum().item()}")
        batch = batch[valid_indices]  # Update batch to match the processed nodes

        # Pass the RNN output through the fully connected layer to produce final predictions
        output = self._fc(rnn_out)  # (batch_size, output_size)

        # Debugging shapes
        print(f"Input time-series length: {x.shape[0]}")
        print(f"RNN output shape: {rnn_out.shape}")
        print(f"Corrected batch shape: {batch.shape}")
        print(f"Final output shape: {output.shape}")

        return output