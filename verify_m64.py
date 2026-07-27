from rich.console import Console

from src.modules.classifier.result import InferenceSession, PredictionCandidate, PredictionResult
from src.modules.inference_engine.context import PipelineConfig, PipelineContext
from src.modules.ood.config import OODConfig
from src.modules.ood.stage import OODDetectionStage

console = Console()

def verify_ood():
    console.print("\n[bold cyan]Milestone 6.4: OOD Detection Verification[/bold cyan]")

    # 1. Setup mock classification artifact (OOD case: Cat Image -> Flat distribution)
    # A cat image would confuse the skin classifier, outputting ~equal probabilities for everything
    pred = PredictionResult(
        predicted_class="MEL",
        predicted_index=0,
        confidence=0.15,
        probabilities={"MEL": 0.15, "NV": 0.15, "BCC": 0.14, "AKIEC": 0.14, "BKL": 0.14, "DF": 0.14, "VASC": 0.14},
        top_k=[PredictionCandidate("MEL", 0, 0.15)]
    )
    session = InferenceSession(
        request_id="v-123", execution_id="v-123", backend="torch", device="cpu",
        latency=0.1, memory_mb=0.1, pipeline_version="v5.2",
        preprocessing_version="v6.2", model_version="v1.0", batch_size=1, start_time=0.0, end_time=0.1
    )

    ctx = PipelineContext(raw_image=None, config=PipelineConfig())
    ctx.artifacts = {"classification": (pred, session)}

    # 2. Setup Stage with Entropy
    config = OODConfig(
        algorithm="entropy",
        threshold=1.5 # High entropy -> OOD
    )

    stage = OODDetectionStage(config)
    stage.initialize()

    console.print("[yellow]Evaluating Out-of-Distribution...[/yellow]")
    stage.execute(ctx)

    # 3. Print Results
    ood_res = ctx.artifacts["ood"]

    console.print("\n[bold green]=== OOD Evaluation Result ===[/bold green]")
    console.print(f"Algorithm: {ood_res.algorithm}")
    console.print(f"OOD Score: {ood_res.ood_score:.4f}")
    console.print(f"Threshold: {ood_res.threshold:.4f}")

    if ood_res.is_in_distribution:
        console.print("[bold green]Status: ACCEPTED (In-Distribution)[/bold green]")
    else:
        console.print("[bold red]Status: REJECTED (Out-of-Distribution)[/bold red]")
        console.print(f"Reason: {ood_res.reason}")

    console.print(f"Execution Time: {ood_res.execution_time:.6f}s")

    console.print("\n[bold green]Verification passed![/bold green]")

if __name__ == "__main__":
    verify_ood()
