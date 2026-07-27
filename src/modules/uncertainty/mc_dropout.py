import numpy as np


def run_mc_dropout(model, x, num_samples=30):
    """Run model with dropout enabled multiple times."""
    import torch
    model.train() # Enable dropout
    with torch.no_grad():
        outputs = []
        for _ in range(num_samples):
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.array(outputs) # (M, N, C)
