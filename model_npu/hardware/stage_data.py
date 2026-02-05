from typing import TypeVar, Generic

T = TypeVar("T")


class StageData(Generic[T]):
    """
    Stage data container with claim-based handshaking.

    A stage prepares data for the next stage. The downstream stage must
    claim the data before the upstream stage can produce new data.
    If data is not claimed, the upstream stage should stall.

    Usage:
        # Upstream stage prepares data
        self.output.prepare(data)

        # Downstream stage claims data
        data = upstream.output.claim()

        # Upstream checks if it should stall
        if self.output.should_stall():
            return  # Don't produce new data
    """

    def __init__(self, empty_value: T) -> None:
        """
        Initialize stage data.

        Args:
            empty_value: The value representing "no data" (e.g., [], None, {})
        """
        self.empty_value = empty_value
        self.data: T = empty_value
        self.valid: bool = False

    def prepare(self, data: T) -> None:
        """
        Prepare data for the downstream stage.

        Should only be called if previous data was claimed.
        """
        self.data = data
        self.valid = self._is_non_empty(data)

    def claim(self) -> T:
        """
        Claim data from the upstream stage.

        Returns the data and marks it as claimed, allowing the
        upstream stage to produce new data.

        Returns:
            The prepared data, or empty_value if nothing is valid.
        """
        if self.valid:
            data = self.data
            self.data = self.empty_value
            self.valid = False
            return data
        return self.empty_value

    def peek(self) -> T:
        """
        Peek at the data without claiming it.

        Returns:
            The prepared data, or empty_value if nothing is valid.
        """
        if self.valid:
            return self.data
        return self.empty_value

    def should_stall(self) -> bool:
        """
        Check if the upstream stage should stall.

        The stage should stall if it has valid data that hasn't been claimed.

        Returns:
            True if the stage should stall.
        """
        return self.valid

    def is_valid(self) -> bool:
        """
        Check if there is valid unclaimed data.

        Returns:
            True if there is valid data waiting to be claimed.
        """
        return self.valid

    def reset(self) -> None:
        """Reset stage data to initial state."""
        self.data = self.empty_value
        self.valid = False

    def _is_non_empty(self, data: T) -> bool:
        """
        Check if data is non-empty.

        Handles common cases: None, empty lists, empty dicts.
        """
        if data is None:
            return False
        if isinstance(data, (list, dict)):
            return len(data) > 0
        return True

    def __repr__(self) -> str:
        return f"StageData(valid={self.valid}, data={self.data})"
