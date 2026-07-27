import numpy as np


def run_ensemble(models, x):
    import torch
    outputs = []
    with torch.no_grad():
        for model in models:
            model.eval()
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.array(outputs) # (M, N, C)
