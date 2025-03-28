"""RNN model implementation."""

from typing import List, Optional
import torch
from torch_geometric.data import Data
from graphnet.models.gnn.gnn import GNN
from graphnet.models.rnn.full_rnn import Full_RNN
from graphnet.utilities.config import save_model_config


class RNN(GNN):
    """The RNN model class.

    A simplified version of RNN_TITO that uses Full_RNN as the backbone
    for processing time-series data and producing final predictions.
    """

    @save_model_config
    def __init__(
        self,
        nb_inputs: int,
        time_series_columns: List[int],
        *,
        rnn_layers: int = 2,
        rnn_hidden_size: int = 64,
        rnn_dropout: float = 0.5,
        features_subset: Optional[List[int]] = None,
        output_size: int = 1,
        embedding_dim: Optional[int] = 0,
    ):
        """Initialize the RNN model.

        Args:
            nb_inputs (int): Number of input features.
            time_series_columns (List[int]): The indices of the input data that
                should be treated as time-series data.
            rnn_layers (int, optional): Number of RNN layers. Defaults to 2.
            rnn_hidden_size (int, optional): Size of the hidden state of the
                RNN. Also determines the size of the output of the RNN.
                Defaults to 64.
            rnn_dropout (float, optional): Dropout to use in the RNN.
                Defaults to 0.5.
            features_subset (List[int], optional): The subset of latent
                features on each node that are used as metric dimensions when
                performing the k-nearest neighbours clustering.
                Defaults to [0,1,2,3]
            output_size (int, optional): Number of output features (e.g.,
                regression targets or scores). Defaults to 1.
            embedding_dim (int, optional): Dimension of the positional embedding.
                Defaults to 0.
        """
        super().__init__(nb_inputs, rnn_hidden_size)
        
        self._hidden_size = rnn_hidden_size
        self._num_layers = rnn_layers
        self._time_series_columns = time_series_columns
        self._embedding_dim = embedding_dim
        self._nb_inputs = nb_inputs
        self._output_size = output_size
        self._features_subset = features_subset

        # Full RNN backbone
        self._full_rnn = Full_RNN(
            nb_inputs=nb_inputs,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            time_series_columns=time_series_columns,
            features_subset=self._features_subset,
            dropout=rnn_dropout,
            output_size=output_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, data: Data) -> torch.Tensor:
        """Apply learnable forward pass of the RNN model.

        Args:
            data: Input graph data.

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        # Pass the data through Full_RNN
        output = self._full_rnn(data)

        # Debugging shapes
        print(f"Output shape: {output.shape}")

        return output