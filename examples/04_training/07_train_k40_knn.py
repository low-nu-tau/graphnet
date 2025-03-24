# Choice of graph representation, architecture, and physics task
from graphnet.models.detector.prometheus import PONETriangle
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodesAsPulses
from graphnet.models.gnn.dynedge import DynEdge
from graphnet.models.task.classification import MulticlassClassificationTask
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

train_dataloader = DataLoader("/mnt/gs21/scratch/robsonj3/k40sim/parquet/K40PulseMap/*", batch_size=16, num_workers=4)
model.fit(train_dataloader=train_dataloader, max_epochs=10)

