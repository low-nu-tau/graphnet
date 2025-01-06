"""Classification-specific `Model` class(es)."""

import torch
from torch import Tensor

from graphnet.models.task import IdentityTask, StandardLearnedTask


class MulticlassClassificationTask(IdentityTask):
    """General task for classifying any number of classes.

    Requires the same number of input features as the number of classes being
    predicted. Returns the untransformed latent features, which are interpreted
    as the logits for each class being classified.
    """


# class BinaryClassifier(Tensor.Module):
#     def __init__(self, input_dim, hidden_layers, output_dim=1, activation="ReLU", loss_fn="BCEWithLogitsLoss"):
#         super(BinaryClassifier, self).__init__()
#         layers = []
#         in_dim = input_dim
#         for h_dim in hidden_layers:
#             layers.append(Tensor.Linear(in_dim, h_dim))
#             layers.append(getattr(Tensor, activation)())
#             in_dim = h_dim
#         layers.append(Tensor.Linear(in_dim, output_dim))
#         self.model = Tensor.Sequential(*layers)
#         self.loss_fn = getattr(Tensor, loss_fn)()
#         self.task = StandardLearnedTask(hidden_size=output_dim, target="target")

#     def forward(self, x):
#         return self.model(x)

#     def compute_loss(self, pred, target):
#         return self.loss_fn(pred, target)

class BinaryClassificationTask(StandardLearnedTask):
    """Performs binary classification."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1
    default_target_labels = ["target"]
    default_prediction_labels = ["target_pred"]

    def _forward(self, x: Tensor) -> Tensor:
        # transform probability of being muon
        return torch.sigmoid(x)


class BinaryClassificationTaskLogits(StandardLearnedTask):
    """Performs binary classification form logits."""

    # Requires one feature, logit for being signal class.
    nb_inputs = 1
    default_target_labels = ["target"]
    default_prediction_labels = ["target_pred"]

    def _forward(self, x: Tensor) -> Tensor:
        return x
