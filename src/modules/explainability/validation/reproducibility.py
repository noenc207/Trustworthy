import hashlib
import os
import random
import sys
from typing import Any

import numpy as np
import torch


class ExecutionEnvironment:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.set_seed(seed)

    def set_seed(self, seed: int):
        self.seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

    def capture_environment(self) -> dict[str, Any]:
        """Captures exact versions, device, and seeds to ensure reproducibility."""
        env = {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "seed": self.seed,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "git_hash": self._get_git_hash(),
            "config_checksum": self._get_config_hash()
        }
        return env

    def _get_git_hash(self) -> str:
        try:
            import subprocess
            res = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
            return res
        except Exception:
            return "unknown_or_not_git"

    def _get_config_hash(self) -> str:
        # Dummy implementation for config hash
        config_str = 'ExplainabilityConfig(primary_algorithm=GRADCAM)'
        return hashlib.sha256(config_str.encode('utf-8')).hexdigest()

