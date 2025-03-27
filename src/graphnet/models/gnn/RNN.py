"""RNN model implementation."""
import torch
from typing import List, Optional

from graphnet.models.gnn.gnn import GNN
from graphnet.models.rnn.full_rnn import Full_RNN  # Import Full_RNN
from graphnet.utilities.config import save_model_config
from torch_geometric.data import Data


class RNN(GNN):
    """The RNN model class.

    Combines the Full_RNN model, intended for data with large
    amounts of DOM activations per event. This model works only with non-
    standard datasets specific to the Full_RNN model. See Full_RNN for more
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
        output_size: int = 1,  # Add output_size for final predictions
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
            output_size (int, optional): Number of output features for final
                predictions. Defaults to 1.
        """
        super().__init__(nb_inputs, rnn_hidden_size)

        # Initialize Full_RNN as the backbone
        self._rnn = Full_RNN(
            nb_inputs=nb_inputs,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            time_series_columns=time_series_columns,
            nb_neighbours=nb_neighbours,
            features_subset=features_subset,
            dropout=rnn_dropout,
            embedding_dim=embedding_dim,
            output_size=output_size,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass of the RNN model."""
        # Forward pass through Full_RNN
        return self._rnn(data)  # Output tensor shape: [batch_size, output_size]