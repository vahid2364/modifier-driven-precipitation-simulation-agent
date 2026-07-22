"""
BaseAgent: interface all agents must implement.
"""

from abc import ABC, abstractmethod
import sys; sys.path.insert(0, ".")
from state import SimState


class BaseAgent(ABC):
    name: str

    @abstractmethod
    def run(self, state: SimState) -> SimState:
        """
        Consume relevant fields from state, do work, return updated state.
        Must set state["status"] and state["error"] appropriately.
        """
        ...
