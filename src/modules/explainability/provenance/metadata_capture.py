import platform
import subprocess
import time
from typing import Any


class MetadataCapture:
    def capture_git_state(self) -> dict[str, str]:
        """Capture current git branch and commit hash."""
        try:
            commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
            return {"commit": commit_hash, "branch": branch}
        except Exception:
            return {"commit": "unknown", "branch": "unknown"}

    def capture_hardware_state(self) -> dict[str, str]:
        """Capture the hardware and OS execution context."""
        return {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        }

    def capture_pipeline_state(self, config: dict[str, Any]) -> dict[str, Any]:
        """Aggregate all metadata for pipeline execution state."""
        return {
            "git": self.capture_git_state(),
            "hardware": self.capture_hardware_state(),
            "config": config,
            "timestamp": str(time.time())
        }
