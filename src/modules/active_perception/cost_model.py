from __future__ import annotations
from .action_space import ObservationAction, ACTION_COSTS

class FixedCostModel:
    """Fixed cost model for active perception actions."""
    def __init__(self, observation_cost: float = 1.0, stop_cost: float = 0.0, abstain_cost: float = 0.0):
        self.observation_cost = observation_cost
        self.stop_cost = stop_cost
        self.abstain_cost = abstain_cost

    def cost(self, action: ObservationAction) -> float:
        """Return the cost for a specific action."""
        if action == ObservationAction.STOP:
            return self.stop_cost
        if action == ObservationAction.ABSTAIN:
            return self.abstain_cost
        return ACTION_COSTS.get(action, self.observation_cost)

    def total_cost(self, history: list[ObservationAction]) -> float:
        """Calculate total cost for a sequence of actions."""
        return sum(self.cost(a) for a in history)
