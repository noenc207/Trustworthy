
from src.modules.explainability.result import OntologyRecord


class ClinicalLiteratureLayer:
    def __init__(self):
        self.literature_db = {
            "melanoma": ["Smith et al. 2020", "Derm Guidelines 2022"]
        }

    def retrieve_references(self, condition: str) -> list[str]:
        return self.literature_db.get(condition.lower(), [])

    def match_ontology(self, condition: str, ontology: OntologyRecord) -> float:
        return 0.9
