from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMetadata:
    class_id: int
    name: str
    icd_code: str
    who_name: str
    alias: str
    risk_level: str
    clinical_priority: int
    description: str
    color: str

class ClassRegistry:
    """Immutable taxonomy registry mapping class IDs to extensive clinical metadata."""

    def __init__(self) -> None:
        self._registry: dict[int, ClassMetadata] = {}

    def register_class(
        self,
        class_id: int,
        name: str,
        icd_code: str,
        who_name: str,
        alias: str,
        risk_level: str,
        clinical_priority: int,
        description: str,
        color: str
    ) -> None:
        if class_id in self._registry:
            raise ValueError(f"Class ID {class_id} is already registered.")

        self._registry[class_id] = ClassMetadata(
            class_id=class_id,
            name=name,
            icd_code=icd_code,
            who_name=who_name,
            alias=alias,
            risk_level=risk_level,
            clinical_priority=clinical_priority,
            description=description,
            color=color
        )

    def get_metadata(self, class_id: int) -> dict:
        if class_id not in self._registry:
            raise KeyError(f"Class ID {class_id} not found in taxonomy.")
        meta = self._registry[class_id]
        return {
            "name": meta.name,
            "icd_code": meta.icd_code,
            "who_name": meta.who_name,
            "alias": meta.alias,
            "risk_level": meta.risk_level,
            "clinical_priority": meta.clinical_priority,
            "description": meta.description,
            "color": meta.color
        }

    def get_name(self, class_id: int) -> str:
        if class_id not in self._registry:
            raise KeyError(f"Class ID {class_id} not found in taxonomy.")
        return self._registry[class_id].name
