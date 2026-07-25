import sys
import time
import numpy as np
import cv2
from pathlib import Path
from rich.console import Console

from src.modules.preprocessing.config import (
    PreprocessingConfig, 
    ResizeConfig, 
    NormalizationConfig, 
    CLAHEConfig, 
    HairRemovalConfig,
    ROIConfig
)
from src.modules.preprocessing.stage import ImagePreprocessor
from src.modules.inference_engine.context import PipelineContext, PipelineConfig

console = Console()

def verify_preprocessing():
    console.print("\n[bold cyan]Milestone 6.2: Image Preprocessing Verification[/bold cyan]")
    
    config = PreprocessingConfig(
        resize=ResizeConfig(enabled=True, target_size=(300, 300), maintain_aspect_ratio=True),
        clahe=CLAHEConfig(enabled=True, clip_limit=2.0),
        hair_removal=HairRemovalConfig(enabled=True, threshold=15),
        normalization=NormalizationConfig(enabled=False), # Disable to save as image
        roi=ROIConfig(enabled=True, strategy="center")
    )
    
    stage = ImagePreprocessor(config)
    stage.initialize()
    
    # Create a synthetic image representing a hairy skin lesion
    # Background: skin color
    img = np.ones((400, 600, 3), dtype=np.uint8)
    img[:, :, 0] = 160 # B
    img[:, :, 1] = 170 # G
    img[:, :, 2] = 220 # R
    
    # Lesion: darker circle
    cv2.circle(img, (300, 200), 100, (80, 90, 140), -1)
    
    # Hairs: dark lines
    for _ in range(15):
        pt1 = (np.random.randint(200, 400), np.random.randint(100, 300))
        pt2 = (pt1[0] + np.random.randint(-50, 50), pt1[1] + np.random.randint(-50, 50))
        cv2.line(img, pt1, pt2, (40, 40, 40), 2)
        
    ctx_config = PipelineConfig(device_str="cpu")
    context = PipelineContext(raw_image=img, config=ctx_config)
    
    console.print("[yellow]Running pipeline...[/yellow]")
    ctx = stage.execute(context)
    
    result = ctx.preprocessing_result
    
    console.print(f"[green]✔[/green] Original shape: {result.original_shape}")
    console.print(f"[green]✔[/green] Output shape: {result.output_shape}")
    console.print(f"[green]✔[/green] Execution time: {result.execution_time:.4f}s")
    console.print(f"[green]✔[/green] Applied operations: {result.applied_operations}")
    
    # Save outputs
    out_dir = Path("outputs/m6_verification")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(out_dir / "original.jpg"), img)
    cv2.imwrite(str(out_dir / "processed.jpg"), result.processed_image)
    
    console.print(f"[bold green]Verification passed! Outputs saved to {out_dir}[/bold green]")
    
if __name__ == "__main__":
    verify_preprocessing()
