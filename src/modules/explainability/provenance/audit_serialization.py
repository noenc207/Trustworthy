import json

from src.modules.explainability.result import EvidenceProvenance


class AuditSerializer:
    def export_checksums(self, provenances: list[EvidenceProvenance], filepath: str) -> None:
        """Export provenance information and checksums to a JSON audit file."""
        data = []
        for prov in provenances:
            data.append({
                "origin": prov.origin,
                "checksum": prov.checksum,
                "timestamp": prov.timestamp,
                "version": prov.version,
                "dependencies": prov.dependencies,
                "source_module": prov.source_module
            })

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
