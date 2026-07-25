import hashlib
import time

from src.modules.explainability.result import EvidenceProvenance


class ProvenanceEngine:
    def create_provenance(self, origin: str, dependencies: list[str], source_module: str, version: str = "1.0.0") -> EvidenceProvenance:
        """Create an EvidenceProvenance object tracking the origin and integrity of evidence."""
        timestamp = str(time.time())
        checksum_content = origin + "".join(dependencies) + timestamp
        checksum = hashlib.sha256(checksum_content.encode('utf-8')).hexdigest()

        return EvidenceProvenance(
            origin=origin,
            dependencies=dependencies,
            version=version,
            checksum=checksum,
            timestamp=timestamp,
            source_module=source_module
        )
