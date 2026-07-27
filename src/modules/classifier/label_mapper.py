
from .exceptions import InvalidInputError


class LabelMapper:
    def __init__(self, mapping: dict[int, str]):
        self.index_to_name = mapping
        self.name_to_index = {v: k for k, v in mapping.items()}

    def index_to_label(self, index: int) -> str:
        if index not in self.index_to_name:
            raise InvalidInputError(f"Unknown label index: {index}", "UNKNOWN_LABEL_INDEX")
        return self.index_to_name[index]

    def label_to_index(self, label: str) -> int:
        if label not in self.name_to_index:
            raise InvalidInputError(f"Unknown label name: {label}", "UNKNOWN_LABEL_NAME")
        return self.name_to_index[label]

    def get_all_class_names(self) -> list[str]:
        return [self.index_to_name[i] for i in range(len(self.index_to_name))]

    @classmethod
    def from_dict(cls, data: dict[int, str]) -> 'LabelMapper':
        return cls(data)
