import time
import uuid
import numpy as np
from pathlib import Path
from typing import Any

from src.modules.classifier.config import ClassifierConfig
from src.modules.classifier.loader import ClassifierLoader
from src.modules.classifier.result import PredictionResult, InferenceSession
from src.modules.classifier.exceptions import UnsupportedBackendError

class InferencePredictor:
    """Orchestrates strategy, backend adapter, and telemetry."""
    
    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.loader = ClassifierLoader()
        
        if self.config.backend.lower() == "torch":
            from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
            self.backend_adapter = TorchBackendAdapter()
        else:
            raise UnsupportedBackendError(f"Unsupported backend: {self.config.backend}")
            
        self.strategy = self.loader.get_strategy(self.config.model_name, self.config.class_names)
        
        if not self.config.lazy_loading:
            self._load_model()
            
    def _get_memory_mb(self) -> float:
        import torch
        import psutil
        import os
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)

    def _load_model(self):
        weights_path = Path(self.config.weights_path)
        self.strategy.load(weights_path, self.backend_adapter)

    def predict(self, image: np.ndarray, request_id: str = None, execution_id: str = None) -> tuple[PredictionResult, InferenceSession]:
        start_time = time.time()
        
        # Ensure model is loaded (handles lazy loading case)
        self._load_model()
        
        # Adapt input for backend
        tensor = self.strategy.adapt_input(image)
        
        # Inference
        raw_output = self.strategy.predict(tensor, self.backend_adapter)
        
        # Postprocess
        result_dict = self.strategy.postprocess_output(raw_output)
        
        end_time = time.time()
        latency = end_time - start_time
        
        prediction = PredictionResult(
            predicted_class=result_dict["predicted_class"],
            predicted_index=result_dict["predicted_index"],
            confidence=result_dict["confidence"],
            probabilities=result_dict["probabilities"],
            top_k=result_dict["top_k"],
            warnings=[]
        )
        
        session = InferenceSession(
            request_id=request_id or str(uuid.uuid4()),
            execution_id=execution_id or str(uuid.uuid4()),
            backend=self.config.backend,
            device=self.config.device,
            latency=latency,
            memory_mb=self._get_memory_mb(),
            pipeline_version="v5.2",
            preprocessing_version="v6.2",
            model_version=self.config.model_version,
            batch_size=1,
            start_time=start_time,
            end_time=end_time
        )
        
        return prediction, session
