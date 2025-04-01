"""Example of training String_RNN model with Binary Classification Task."""

import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.gnn import String_RNN
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodeAsDOMTimeSeries
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset, ParquetDataset

# Constants
features = FEATURES.PROMETHEUS
truth = TRUTH.PROMETHEUS


def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Train the String_RNN model with Binary Classification Task."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="string_rnn_training",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )
    else:
        wandb_logger = None

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        "dataset_reference": (
            SQLiteDataset if path.endswith(".db") else ParquetDataset
        ),
    }

    # Define the graph structure
    graph_definition = KNNGraph(
        nb_nearest_neighbours=8,
        node_definition=NodeAsDOMTimeSeries(),
    )

    # Use GraphNetDataModule to load in data
    datamodule = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "truth": truth,
            "truth_table": truth_table,
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": [config["pulsemap"]],
            "path": config["path"],
            "index_column": "event_no",
            "labels": {
                "binary_target": target,
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

    # Initialize the String_RNN model
    backbone = String_RNN(
        nb_inputs=len(features),
        hidden_size=64,
        num_layers=2,
        time_series_columns=[0, 1],  # Example: charge and time columns
        dropout=0.5,
        embedding_dim=16,
    )

    # Add Binary Classification Task
    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        prediction_labels="binary_prediction",
        loss_function=BinaryCrossEntropyLoss(),
    )

    # Combine GNN and Task into a StandardModel
    model = StandardModel(
        graph_definition=graph_definition,
        backbone=backbone,
        tasks=[task],
        optimizer_class=Adam,
        optimizer_kwargs={"lr": 1e-03, "eps": 1e-03},
        scheduler_class=ReduceLROnPlateau,
        scheduler_kwargs={
            "patience": config["early_stopping_patience"],
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
        },
    )

    # Training model
    model.fit(
        datamodule.train_dataloader,
        datamodule.val_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Get predictions
    additional_attributes = [
        "event_no",
    ]
    prediction_columns = [
        "binary_prediction",
    ]

    results = model.predict_as_dataframe(
        datamodule.val_dataloader,
        additional_attributes=additional_attributes,
        prediction_columns=prediction_columns,
        gpus=config["fit"]["gpus"],
    )

    # Save predictions and model to file
    archive = os.path.join(EXAMPLE_OUTPUT_DIR, "train_String_RNN_model")
    run_name = "String_RNN_{}_example".format(config["target"])
    db_name = path.split("/")[-1].split(".")[0]
    save_path = os.path.join(archive, db_name, run_name)
    logger.info(f"Writing results to {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # Save results as .csv
    results.to_csv(f"{save_path}/results.csv")

    # Save full model (including weights) to .pth file - Not version proof
    model.save(f"{save_path}/model.pth")

    # Save model config and state dict - Version safe save method
    model.save_state_dict(f"{save_path}/state_dict.pth")
    model.save_config(f"{save_path}/model_config.yml")


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train String_RNN model with Binary Classification Task using StandardModel.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"{EXAMPLE_DATA_DIR}/sqlite/prometheus/prometheus-events.db",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="total",
    )

    parser.add_argument(
        "--target",
        help=(
            "Name of feature to use as regression target (default: "
            "%(default)s)"
        ),
        default="binary_target",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="mc_truth",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 1),
        ("early-stopping-patience", 2),
        ("batch-size", 16),
        "num-workers",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args, unknown = parser.parse_known_args()

    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.wandb,
    )