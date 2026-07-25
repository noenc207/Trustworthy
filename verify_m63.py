import sys
import time
import numpy as np
import torch
from pathlib import Path
from rich.console import Console

from src.modules.classifier.config import ClassifierConfig
from src.modules.classifier.stage import LesionClassificationStage
from src.modules.inference_engine.context import PipelineContext, PipelineConfig
from src.modules.preprocessing.results import PreprocessingResult
from src.domain.taxonomy.registry import ClassRegistry

console = Console()

def create_dummy_model(path: Path):
    """Creates a lightweight dummy torch model for testing inference."""
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            # Input is (N, C, H, W)
            # Output should be (N, 7) for 7 classes
            batch_size = x.shape[0]
            # Just return random logits
            return torch.randn(batch_size, 7)
            
    model = DummyModel()
    scripted = torch.jit.script(model)
    scripted.save(str(path))

def verify_classification():
    console.print("\n[bold cyan]Milestone 6.3: Lesion Classification Verification[/bold cyan]")
    
    # 1. Initialize Taxonomy
    registry = ClassRegistry()
    registry.register_class(0, "MEL", "C43.9", "Melanoma", "Melanoma", "High", 1, "Malignant melanoma", "#FF0000")
    
    # 2. Setup dummy weights
    out_dir = Path("outputs/m63_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    dummy_path = out_dir / "dummy_effnet.pt"
    create_dummy_model(dummy_path)
    
    # 3. Setup Stage
    config = ClassifierConfig(
        model_name="efficientnet_v2",
        backend="torch",
        device="cpu",
        weights_path=str(dummy_path),
        lazy_loading=True
    )
    
    stage = LesionClassificationStage(config)
    stage.initialize()
    
    # 4. Mock PipelineContext with preprocessing artifact
    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {}
    ctx.artifacts["preprocessing"] = PreprocessingResult(
        processed_image=np.random.rand(224, 224, 3).astype(np.float32),
        original_shape=(400, 400),
        output_shape=(224, 224)
    )
    
    console.print("[yellow]Executing Pipeline Stage...[/yellow]")
    stage.execute(ctx)
    
    # 5. Extract results
    pred, session = ctx.artifacts["classification"]
    
    console.print("\n[bold green]=== Inference Session ===[/bold green]")
    console.print(f"Request ID: {session.request_id}")
    console.print(f"Device: {session.device}")
    console.print(f"Backend: {session.backend}")
    console.print(f"Latency: {session.latency:.4f}s")
    console.print(f"Pipeline Ver: {session.pipeline_version}")
    
    console.print("\n[bold green]=== Prediction Result ===[/bold green]")
    console.print(f"Predicted Class: {pred.predicted_class} (Index: {pred.predicted_index})")
    console.print(f"Confidence: {pred.confidence:.2%}")
    
    console.print("\n[bold green]Top-5 Predictions:[/bold green]")
    for rank, candidate in enumerate(pred.top_k, 1):
        console.print(f"{rank}. {candidate.class_name}: {candidate.probability:.2%}")
        
    console.print(f"\n[bold green]Verification passed![/bold green]")
    
if __name__ == "__main__":
    verify_classification()
