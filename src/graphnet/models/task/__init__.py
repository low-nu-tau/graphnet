"""Physics task-specific modules to be used as model "read-outs"."""

from .task import (
    Task,
    IdentityTask,
    StandardLearnedTask,
    StandardFlowTask,
)
from .classification import (
    MulticlassClassificationTask,
    # BinaryClassifier,
    BinaryClassificationTask,
    BinaryClassificationTaskLogits,
)
