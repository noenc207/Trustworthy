from __future__ import annotations
from enum import Enum

class ObservationAction(Enum):
    KEEP_FULL = 0
    ZOOM_CENTER = 1
    ZOOM_BORDER = 2
    LEFT_REGION = 3
    RIGHT_REGION = 4
    TOP_REGION = 5
    BOTTOM_REGION = 6
    PIGMENT_REGION = 7
    TEXTURE_REGION = 8
    COLOR_NORMALIZED = 9
    ARTIFACT_SUPPRESSED = 10
    STOP = 11
    ABSTAIN = 12

ACTION_COSTS: dict[ObservationAction, float] = {
    ObservationAction.KEEP_FULL: 1.0,
    ObservationAction.ZOOM_CENTER: 1.0,
    ObservationAction.ZOOM_BORDER: 1.0,
    ObservationAction.LEFT_REGION: 1.0,
    ObservationAction.RIGHT_REGION: 1.0,
    ObservationAction.TOP_REGION: 1.0,
    ObservationAction.BOTTOM_REGION: 1.0,
    ObservationAction.PIGMENT_REGION: 1.0,
    ObservationAction.TEXTURE_REGION: 1.0,
    ObservationAction.COLOR_NORMALIZED: 1.0,
    ObservationAction.ARTIFACT_SUPPRESSED: 1.0,
    ObservationAction.STOP: 0.0,
    ObservationAction.ABSTAIN: 0.0,
}

OBSERVATION_ACTIONS: list[ObservationAction] = [
    a for a in ObservationAction if a not in (ObservationAction.STOP, ObservationAction.ABSTAIN)
]

def get_valid_actions(history: list[ObservationAction], budget_remaining: int) -> list[ObservationAction]:
    """Get list of valid actions based on history and budget."""
    valid_actions = []
    if budget_remaining > 0:
        for action in OBSERVATION_ACTIONS:
            if action not in history:
                valid_actions.append(action)
    valid_actions.append(ObservationAction.STOP)
    valid_actions.append(ObservationAction.ABSTAIN)
    return valid_actions
