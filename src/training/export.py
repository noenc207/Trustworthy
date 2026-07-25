"""Model export module for training engine."""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, Dict, Optional

def export_torchscript(model: nn.Module, save_path: Path, input_shape: Tuple = (1, 3, 224, 224)) -> Path:
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(model, dummy_input)
        
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(save_path_obj))
    
    return save_path_obj

def export_onnx(model: nn.Module, save_path: Path, input_shape: Tuple = (1, 3, 224, 224), dynamic_axes: Optional[Dict] = None, opset_version: int = 17) -> Path:
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)
    
    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    if dynamic_axes is None:
        dynamic_axes = {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path_obj),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )
        
    return save_path_obj

def verify_export(original_model: nn.Module, exported_path: Path, input_shape: Tuple) -> bool:
    original_model.eval()
    device = next(original_model.parameters()).device
    dummy_input = torch.randn(input_shape, device=device)
    
    with torch.no_grad():
        original_output = original_model(dummy_input)
        
    exported_path_str = str(exported_path)
    
    if exported_path_str.endswith('.pt') or exported_path_str.endswith('.pth'):
        traced_model = torch.jit.load(exported_path_str, map_location=device)
        traced_model.eval()
        with torch.no_grad():
            traced_output = traced_model(dummy_input)
            
        return torch.allclose(original_output, traced_output, rtol=1e-3, atol=1e-5)
        
    elif exported_path_str.endswith('.onnx'):
        try:
            import onnxruntime as ort
            ort_session = ort.InferenceSession(exported_path_str)
            ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
            ort_outs = ort_session.run(None, ort_inputs)
            
            return torch.allclose(original_output.cpu(), torch.from_numpy(ort_outs[0]), rtol=1e-3, atol=1e-5)
        except ImportError:
            print("onnxruntime not installed. Cannot verify ONNX export.")
            return False
            
    return False
