# Choice of graph representation, architecture, and physics task
from graphnet.models.detector.pone import PONE
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.data.dataloader import DataLoader
from torch.utils.data import random_split
from graphnet.data.dataset.dataset import EnsembleDataset

from graphnet.constants import EXAMPLE_OUTPUT_DIR, TEST_DATA_DIR


# Choice of loss function and Model class
from graphnet.training.loss_functions import MAELoss
from graphnet.models import StandardModel

import torch
from torch_geometric.data import Data
from graphnet.training.labels import Label
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from graphnet.utilities.logging import Logger
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import rank_zero_only
import os
import time
import random
import wandb

logger = Logger()

class MyCustomLabel(Label):
    """Class for producing my label."""
    def __init__(self):
        """Construct `MyCustomLabel`."""
        # Base class constructor
        super().__init__(key="my_custom_label")

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        muon_event = ... # If the event comes from file containing GenerateSingleMuons, set to 1, otherwise set to 0
        return muon_event


class WandbMetricsLogger(Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        batch_time = time.time() - self.batch_start_time
        metrics = trainer.callback_metrics
        wandb.log({
            "batch_idx": batch_idx,
            "train_loss_batch": metrics.get("train_loss", None),
            "batch_time": batch_time,  # Log batch processing time
        })


graph_definition = KNNGraph(
    detector=PONE(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
    input_feature_names=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],  # Define features explicitly
)


signal = ParquetDataset(
    #path= f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone",
    path= "/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/pone_script_test",
    pulsemaps="PMTResponse_nonoise",
    truth_table="GenerateSingleMuons_39_pmtsim_pframe_truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["pid"],
    graph_definition = graph_definition,
)

background = ParquetDataset(
    #path= f"{EXAMPLE_OUTPUT_DIR}/convert_i3_files/pone",
    path="/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/pone_script_test",
    pulsemaps="K40PulseMap",
    truth_table="K40PulseMap_truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["pid"],
    graph_definition = graph_definition,
)

print("Signal length: ", len(signal))
print("Signal features: ", signal._features)
print("Graph definition input features: ", graph_definition._input_feature_names)

#since background is way larger we want to subsample it
generator1 = torch.Generator().manual_seed(42)
subsampled_bkg, _ = random_split(background, [10, len(background) - 10], generator=generator1)
print("Subsampled_background: ", len(subsampled_bkg))

subsampled_signal, _ = random_split(signal, [10, len(signal) - 10], generator=generator1)
print("Signal_Subsampled: ", len(subsampled_signal))

# create the total dataset from now equally sized bkg and signal datasets
ensemble_dataset = EnsembleDataset([subsampled_signal, subsampled_bkg]) # change: subsampled_signal to signal
# ensemble_dataset.add_label(MyCustomLabel())

# and now we can do the split in train, val, test
train_set, val_set, test_set  = random_split(ensemble_dataset, [0.8, 0.1, 0.1], generator=generator1)


train_dataloader = DataLoader(train_set, batch_size=1, num_workers=1)
validate_dataloader = DataLoader(val_set, batch_size=1, num_workers=1)
test_dataloader = DataLoader(test_set, batch_size=1, num_workers=1)

#check the lengths of the loaders
print(len(train_dataloader))
print(len(validate_dataloader))
print(len(test_dataloader))

#Check batch size
print(train_dataloader.batch_size)
print(validate_dataloader.batch_size)
print(test_dataloader.batch_size)

# Configuring the components

# Represents the data as a point-cloud graph where each
# node represents a pulse of Cherenkov radiation
# edges drawn to the 8 nearest neighbours

backbone = DynEdge(
    nb_inputs=graph_definition.nb_outputs,
    global_pooling_schemes=["min", "max", "mean"],
)
task = MulticlassClassificationTask(
    hidden_size=backbone.nb_outputs,
    nb_outputs=backbone.nb_outputs,
    target_labels="multiclass_classification",
    loss_function=MAELoss(),
)

# Construct the Model with GPU settings passed via trainer_kwargs
model = StandardModel(
    graph_definition=graph_definition,
    backbone=backbone,
    tasks=[task],
)

# Initialize wandb
wandb_run = wandb.init(
    project="array-performance-1",
    config={
        "learning_rate": 0.02,
        "epochs": 1,
    },
)

# Add the WandbMetricsLogger callback
wandb_logger_callback = WandbMetricsLogger()

for i in range(5):
    print(train_set[i])  # Check if the dataset returns valid samples


batch = next(iter(train_dataloader))
preds = model(batch)  # Check if this runs without errors
print(preds)

# Train the model with the callback
model.fit(
    train_dataloader,
    max_epochs=wandb_run.config["epochs"],
    callbacks=[wandb_logger_callback],  # Add the callback here
)

print("TRAIN MODEL HAS FINISHED")

# Predict and save results
results = model.predict_as_dataframe(
    dataloader=test_dataloader,
    additional_attributes=model.target_labels + ["event_no"],
)
# Save predictions and model to file
""" outdir = "/mnt/home/robsonj3/knn_output"
os.makedirs(outdir, exist_ok=True)
results.to_csv(f"{outdir}/results.csv")
model.save_state_dict(f"{outdir}/state_dict.pth") """
# model.save(f"{outdir}/model.pth")