"""Class(es) for constructing training labels at runtime."""

from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data
from graphnet.utilities.logging import Logger


class Label(ABC, Logger):
    """Base `Label` class for producing labels from single `Data` instance."""

    def __init__(self, key: str):
        """Construct `Label`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
        """
        self._key = key

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    @property
    def key(self) -> str:
        """Return value of `key`."""
        return self._key

    @abstractmethod
    def __call__(self, graph: Data) -> torch.tensor:
        """Label-specific implementation."""

class test_icecube(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "test",
        dom_pos_x: str = "dom_x",
        dom_pos_y: str = "dom_y",
        dom_pos_z: str = "dom_z",
    ):
        """Construct `test`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            dom_pos_x: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            dom_pos_y: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
            dom_pos_z: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._dom_x = dom_pos_x
        self._dom_y = dom_pos_y
        self._dom_z = dom_pos_z

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        print(graph[self._dom_x])
        x = graph[self._dom_x]
        y = graph[self._dom_y]
        z = graph[self._dom_z]
        return torch.cat((x, y, z), dim=1)

class Direction(Label):
    """Class for producing particle direction/pointing label."""

    def __init__(
        self,
        key: str = "direction",
        azimuth_key: str = "azimuth",
        zenith_key: str = "zenith",
    ):
        """Construct `Direction`.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            azimuth_key: The name of the pre-existing key in `graph` that will
                be used to access the azimiuth angle, used when calculating
                the direction.
            zenith_key: The name of the pre-existing key in `graph` that will
                be used to access the zenith angle, used when calculating the
                direction.
        """
        self._azimuth_key = azimuth_key
        self._zenith_key = zenith_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        x = torch.cos(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        y = torch.sin(graph[self._azimuth_key]) * torch.sin(
            graph[self._zenith_key]
        ).reshape(-1, 1)
        z = torch.cos(graph[self._zenith_key]).reshape(-1, 1)
        return torch.cat((x, y, z), dim=1)


class Track(Label):
    """Class for producing NuMuCC label.

    Label is set to `1` if the event is a NuMu CC event, else `0`.
    """

    def __init__(
        self,
        key: str = "track",
        pid_key: str = "pid",
        interaction_key: str = "interaction_type",
    ):
        """Construct `Track` label.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            pid_key: The name of the pre-existing key in `graph` that will
                be used to access the pdg encoding, used when calculating
                the direction.
            interaction_key: The name of the pre-existing key in `graph` that
                will be used to access the interaction type (1 denoting CC),
                used when calculating the direction.
        """
        self._pid_key = pid_key
        self._int_key = interaction_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        is_numu = torch.abs(graph[self._pid_key]) == 14
        is_cc = graph[self._int_key] == 1
        return (is_numu & is_cc).type(torch.int)

class DoublePulse(Label):
    """Class for producing NuMuCC label.

    Label is set to `1` if the event is a NuMu CC event, else `0`.
    """

    def __init__(
        self,
        key: str = "doublepulse",
        pid_key: str = "pid",
        interaction_key: str = "interaction_type",
    ):
        """Construct `Track` label.

        Args:
            key: The name of the field in `Data` where the label will be
                stored. That is, `graph[key] = label`.
            pid_key: The name of the pre-existing key in `graph` that will
                be used to access the pdg encoding, used when calculating
                the direction.
            interaction_key: The name of the pre-existing key in `graph` that
                will be used to access the interaction type (1 denoting CC),
                used when calculating the direction.
        """
        self._pid_key = pid_key
        self._int_key = interaction_key

        # Base class constructor
        super().__init__(key=key)

    def __call__(self, graph: Data) -> torch.tensor:
        """Compute label for `graph`."""
        is_tau = torch.abs(graph[self._pid_key]) == 16
        is_cc = graph[self._int_key] == 1
        return (is_tau & is_cc).type(torch.int)