from abc import ABC, abstractmethod
from typing import List, Dict, Any

class RoutingStrategy(ABC):
    @abstractmethod
    async def select_backend(self, request: Dict[str, Any], backends: List[str], system_state: Dict[str, Any]) -> str:
        """
        Selects the best backend for the given request.
        :param request: The request payload (prompt, model, etc.)
        :param backends: List of backend URLs/IDs
        :param system_state: Current state (queue lengths, loads)
        :return: Selected backend URL/ID
        """
        pass


