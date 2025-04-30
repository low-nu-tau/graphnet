# Import necessary modules
from typing import Any, Dict, List, Optional

from graphnet.models import StandardModel
from graphnet.models.detector.icecube import IceCube86
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.gnn import RNN_TITO

from graphnet.models.task.reconstruction import (
    DirectionReconstructionWithKappa,
)
from graphnet.training.labels import Direction
from graphnet.training.loss_functions import VonMisesFisher3DLoss

# Example training syntax
from graphnet.data import GraphNeTDataModule
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR, TEST_PARQUET_DATA
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset


features = FEATURES.ICECUBE86
truth = TRUTH.ICECUBE86
path = TEST_PARQUET_DATA

# Configuration
config: Dict[str, Any] = {
    "path": path,
    "pulsemap": "SRTInIcePulses",
    "batch_size": 10,
    "num_workers": 10,
    "target": "direction",
    "early_stopping_patience": 2,
    "fit": {
        "gpus": [0,1],
        "max_epochs": 1,
    },
    "dataset_reference": (
        SQLiteDataset if path.endswith(".db") else ParquetDataset
    ),
}

# Hardcoded GraphDefinition for IceCube-86
graph_definition = KNNGraph(
    detector=IceCube86(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,  # Hardcoded number of neighbors
)

# Hardcoded Backbone (e.g., DynEdge)
backbone = RNN_TITO(
    nb_inputs=graph_definition.nb_outputs,
    nb_neighbours=8,
    time_series_columns=[4, 3],
    rnn_layers=2,
    rnn_hidden_size=64,
    rnn_dropout=0.5,
    features_subset=[0, 1, 2, 3],
    dyntrans_layer_sizes=[(256, 256), (256, 256), (256, 256), (256, 256)],
    post_processing_layer_sizes=[336, 256],
    readout_layer_sizes=[256, 128],
    global_pooling_schemes=["max"],
    embedding_dim=0,
    n_head=16,
    use_global_features=True,
    use_post_processing_layers=True,
)

# Hardcoded Task (e.g., Energy Reconstruction)

task = DirectionReconstructionWithKappa(
    hidden_size=backbone.nb_outputs,
    target_labels=config["target"],
    loss_function=VonMisesFisher3DLoss(),
)

# Construct the hardcoded Model
model = StandardModel(
    graph_definition=graph_definition,
    backbone=backbone,
    tasks=[task],
)


# Use GraphNetDataModule to load in data
dm = GraphNeTDataModule(
    dataset_reference=config["dataset_reference"],
    dataset_args={
        "truth": truth,
        "truth_table": "mc_truth",
        "features": features,
        "graph_definition": graph_definition,
        "pulsemaps": [config["pulsemap"]],
        "path": config["path"],
        "index_column": "event_no",
        "labels": {
            "direction": Direction(
                azimuth_key="injection_azimuth",
                zenith_key="injection_zenith",
            )
        },
    },
    train_dataloader_kwargs={
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
    },
    test_dataloader_kwargs={
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
    },
)

training_dataloader = dm.train_dataloader
model.fit(train_dataloader=training_dataloader, max_epochs=10)