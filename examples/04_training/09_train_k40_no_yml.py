# Choice of graph representation, architecture, and physics task
from graphnet.models.detector.prometheus import PONETriangle
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.classification import MulticlassClassificationTask
from graphnet.data.dataloader import DataLoader
from graphnet.data.dataset.parquet.parquet_dataset import ParquetDataset
from graphnet.data.dataloader import DataLoader

# Choice of loss function and Model class
from graphnet.training.loss_functions import MAELoss
from graphnet.models import StandardModel

# Configuring the components

# Represents the data as a point-cloud graph where each
# node represents a pulse of Cherenkov radiation
# edges drawn to the 8 nearest neighbours

graph_definition = KNNGraph(
    detector=PONETriangle(),
    node_definition=NodesAsPulses(),
    nb_nearest_neighbours=8,
)
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

train_dataset = ParquetDataset(
    path="/mnt/gs21/scratch/robsonj3/k40sim/parquet/k40_merged_parquet_train",
    pulsemaps="K40PulseMap",
    truth_table="mc_truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=[],
    graph_definition = graph_definition,
)

valid_dataset = ParquetDataset(
    path="/mnt/gs21/scratch/robsonj3/k40sim/parquet/k40_merged_parquet_validate",
    pulsemaps="K40PulseMap",
    truth_table="mc_truth",
    features=["dom_x", "dom_y", "dom_z", "dom_time", "charge"],
    truth=[],
    graph_definition = graph_definition,
)

train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=10)
validate_dataloader = DataLoader(valid_dataset, batch_size=128, num_workers=10)

# Train model
model.fit(train_dataloader=train_dataloader, max_epochs=10)

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
