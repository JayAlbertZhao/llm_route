from .base import RoutingStrategy
import itertools
from typing import List, Dict, Any

class RoundRobinStrategy(RoutingStrategy):
    def __init__(self):
        self._iterator = None
        self._last_backend_list = []

    async def select_backend(self, request: Dict[str, Any], backends: List[str], system_state: Dict[str, Any]) -> str:
        if not backends:
            raise ValueError("No backends available")
        
        # If backend list changes, reset iterator
        if backends != self._last_backend_list:
            self._iterator = itertools.cycle(backends)
            self._last_backend_list = list(backends) # Copy
            
        return next(self._iterator)


