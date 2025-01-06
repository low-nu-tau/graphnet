import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from utils import cluster_and_pad
# from graphnet.models.task import BinaryClassifier
from graphnet.models.rnn.node_rnn import Node_RNN
from torch_geometric.data import Data

"""
V. Parrish Dec 2024
This code is a goddamn mess. Please do NOT look beyond it. My goal is I want to use the Node_RNN model + some binary classification, but the way that's defined in this framework is as a task? Which is weird. I also know that the training loop at the model is lame. I need to really restructure this to look more like example 05. 

That is my goal for next time... anyway, here's Wonderwall

NOTE: so many things... uh
1. i am obviously not using any path to samples rn and because of that
2. all of my features and that fun stuff is literally meaningless
3. but the general gist is this... so it is where we can start from

"""
# Configuration
config = {
    "path": "path/to/data",
    "batch_size": 32,
    "num_workers": 4,
    "max_epochs": 10,
    "learning_rate": 0.001,
    "hidden_layers": [64, 32],
    "input_dim": 6,
    "rnn_hidden_dim": 128,
    "rnn_num_layers": 2,
    "rnn_output_dim": 64,
    "time_series_columns": [0, 1, 2],  # Example columns for time series
}

# Example signal and background data (replace with your actual data)
signal_data = np.random.rand(100, config["input_dim"])  # 100 signal samples, input_dim features
background_data = np.random.rand(100, config["input_dim"])  # 100 background samples, input_dim features

# Create labels for signal (1) and background (0)
signal_labels = np.ones(100)
background_labels = np.zeros(100)

# Combine signal and background data
X = np.vstack((signal_data, background_data))
y = np.hstack((signal_labels, background_labels))

# Specify the columns to be used for clustering
cluster_columns = [0, 1, 2]

# Initialize the cluster_and_pad class
cluster_class = cluster_and_pad(x=X, cluster_columns=cluster_columns)

# Add percentile summaries for specific columns
cluster_class.add_percentile_summary(summarization_indices=[3, 4, 5], percentiles=[10, 50, 90])

# Add standard deviation for a specific column
cluster_class.add_std(columns=[4])

# Retrieve the clustered data with all aggregate statistics
clustered_data = cluster_class.clustered_x

# Create a DataLoader for training
train_dataset = TensorDataset(torch.tensor(clustered_data, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])

# Initialize the NodeRNN model
rnn_model = Node_RNN(
    nb_inputs=config["input_dim"],
    hidden_size=config["rnn_hidden_dim"],
    num_layers=config["rnn_num_layers"],
    time_series_columns=config["time_series_columns"],
    nb_neighbours=8,
    features_subset=[0, 1, 2, 3],
    dropout=0.5,
    embedding_dim=0
)

# Initialize the BinaryClassifier model
classifier_model = BinaryClassifier(
    input_dim=config["rnn_hidden_dim"],
    hidden_layers=config["hidden_layers"],
    output_dim=1,
    activation="ReLU",
    loss_fn="BCEWithLogitsLoss"
)

# Define the optimizer
optimizer = Adam(list(rnn_model.parameters()) + list(classifier_model.parameters()), lr=config["learning_rate"])

# Training loop
for epoch in range(config["max_epochs"]):
    rnn_model.train()
    classifier_model.train()
    epoch_loss = 0
    for batch in train_loader:
        x, y = batch
        optimizer.zero_grad()
        
        # Convert batch to Data object for Node_RNN
        data = Data(x=x, y=y)
        
        # Process data with Node_RNN
        rnn_output = rnn_model(data)
        
        # Extract features for classifier
        rnn_features = rnn_output.x
        
        # Classify using the BinaryClassifier
        y_hat = classifier_model(rnn_features).squeeze()
        loss = classifier_model.compute_loss(y_hat, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{config['max_epochs']}, Loss: {epoch_loss / len(train_loader)}")

# Example data for prediction (replace with your actual data)
X_test = np.random.rand(20, config["input_dim"])  # 20 samples, input_dim features

# Cluster and pad the test data
cluster_class_test = cluster_and_pad(x=X_test, cluster_columns=cluster_columns)
cluster_class_test.add_percentile_summary(summarization_indices=[3, 4, 5], percentiles=[10, 50, 90])
cluster_class_test.add_std(columns=[4])
clustered_test_data = cluster_class_test.clustered_x

# Convert test data to Data object for Node_RNN
test_data = Data(x=torch.tensor(clustered_test_data, dtype=torch.float32))

# Make predictions
rnn_model.eval()
classifier_model.eval()
with torch.no_grad():
    rnn_output_test = rnn_model(test_data)
    rnn_features_test = rnn_output_test.x
    predictions = classifier_model(rnn_features_test).squeeze()

print(predictions)