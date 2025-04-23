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

# Choice of loss function and Model class
from graphnet.training.loss_functions import MAELoss
from graphnet.models import StandardModel

import torch
from torch_geometric.data import Data
from graphnet.training.labels import Label

class MyCustomLabel(Label):
    """Class for producing my label."""
    def __init__(self):
        """Construct `MyCustomLabel`."""
        # Base class constructor
        super().__init__(key="my_custom_label")

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        label = ...  # Your computations here.
        return label
    
graph_definition = KNNGraph(
    detector=PONE(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
)

signal = ParquetDataset(
    path="/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/sim/jacob_test_train/truth",
    pulsemaps="truth", # PulseMap_NoNoise
    truth_table="truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["zenith", "azimuth", "energy"],
    graph_definition = graph_definition,
)

print("Signal length: ", len(signal))

background = ParquetDataset(
    path="/mnt/research/IceCube/PONE/jp_pone_sim/k40sim/sim/jacob_test_train/K40PulseMap",
    pulsemaps="K40PulseMap",
    truth_table="truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["zenith", "azimuth", "energy"],
    graph_definition = graph_definition,
)

print("Background length: ", len(signal))

#since background is way larger we want to subsample it
generator1 = torch.Generator().manual_seed(42)
subsampled_bkg, _  = random_split(background, [0.25, 0.75], generator=generator1)
print("Subsampled_background: ", len(subsampled_bkg))

# create the total dataset from now equally sized bkg and signal datasets
ensemble_dataset = EnsembleDataset([signal, subsampled_bkg])

# and now we can do the split in train, val, test
train_set, val_set, test_set  = random_split(background, [0.8, 0.1, 0.1], generator=generator1)


train_dataloader = DataLoader(train_set, batch_size=128, num_workers=10)
validate_dataloader = DataLoader(val_set, batch_size=128, num_workers=10)
test_dataloader = DataLoader(test_set, batch_size=128, num_workers=10)

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

# Construct the Model
model = StandardModel(
    graph_definition=graph_definition,
    backbone=backbone,
    tasks=[task],
)

model.fit(ensemble_dataset, max_epochs=10)

""" train_dataset = ParquetDataset(
    path="/mnt/home/robsonj3/graphnet/data/tests/parquet/jacob_test_train",
    pulsemaps="K40PulseMap",
    truth_table="truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=["zenith", "azimuth", "energy"],
    graph_definition = graph_definition,
)

train_dataloader = DataLoader(train, batch_size=128, num_workers=10) """

# Train model
# model.fit(train_dataloader=train_dataloader, max_epochs=10)

print("TRAIN MODEL HAS FINISHED")
""" results = model.predict_as_dataframe(
    test_dataloader=test_dataloader,
    additional_attributes=model.target_labels + ["event_no"],
) """

# Save predictions and model to file
""" outdir = "/mnt/home/robsonj3/knn_output"
os.makedirs(outdir, exist_ok=True)
results.to_csv(f"{outdir}/results.csv")
model.save_state_dict(f"{outdir}/state_dict.pth")
model.save(f"{outdir}/model.pth") """
