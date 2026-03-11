from abc import ABC, abstractmethod


class Module(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def tick(self) -> None:
        pass
