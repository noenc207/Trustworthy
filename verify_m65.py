import numpy as np
from rich.console import Console

from src.modules.classifier.result import PredictionCandidate, PredictionResult
from src.modules.inference_engine.context import PipelineConfig, PipelineContext
from src.modules.uncertainty.config import UncertaintyConfig
from src.modules.uncertainty.stage import UncertaintyEstimationStage

console = Console()

def verify_uncertainty():
    console.print("\n[bold cyan]Milestone 6.5: Uncertainty Estimation Verification[/bold cyan]")

    # 1. Setup mock classification artifact
    pred = PredictionResult(
        predicted_class="MEL",
        predicted_index=0,
        confidence=0.85,
        probabilities={"MEL": 0.85, "NV": 0.10, "BCC": 0.05},
        top_k=[PredictionCandidate("MEL", 0, 0.85)]
    )

    # 2. Mock stochastic forward passes for MC Dropout (30 samples)
    np.random.seed(123)
    base_probs = np.array([0.85, 0.10, 0.05])
    samples = np.clip(base_probs + np.random.normal(0, 0.15, (30, 3)), 1e-9, 1)
    samples = samples / samples.sum(axis=1, keepdims=True)

    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {
        "classification": (pred, None),
        "stochastic_samples": samples
    }

    # 3. Setup Stage with MC Dropout
    config = UncertaintyConfig(
        algorithm="mc_dropout",
        threshold=0.5, # Epistemic uncertainty limit
        sampling_count=30
    )

    stage = UncertaintyEstimationStage(config)
    stage.initialize()

    console.print("[yellow]Evaluating Predictive Uncertainty...[/yellow]")
    stage.execute(ctx)

    # 4. Print Results
    unc = ctx.artifacts["uncertainty"]

    console.print("\n[bold green]=== Uncertainty Evaluation Result ===[/bold green]")
    console.print(f"Algorithm: {unc.algorithm}")
    console.print(f"Sampling Count: {unc.metadata.get('sampling_count', 1)}")
    console.print(f"Predictive Entropy (Total): {unc.predictive_entropy:.4f}")
    console.print(f"Epistemic Uncertainty: {unc.epistemic_uncertainty:.4f}")
    console.print(f"Aleatoric Uncertainty: {unc.aleatoric_uncertainty:.4f}")
    console.print(f"95% Confidence Interval: [{unc.confidence_interval[0]:.2%}, {unc.confidence_interval[1]:.2%}]")

    if unc.is_reliable:
        console.print("[bold green]Status: RELIABLE[/bold green]")
    else:
        console.print("[bold red]Status: UNRELIABLE[/bold red]")
        console.print(f"Reason: {unc.reason}")

    console.print(f"Execution Time: {unc.execution_time:.6f}s")

    console.print("\n[bold green]Verification passed![/bold green]")

if __name__ == "__main__":
    verify_uncertainty()
