import numpy as np

def fuse_uncertainties(entropy, mutual_information, variation_ratio):
    # Simple standardized fusion
    # In practice, these would be normalized by their max bounds
    return float(np.mean([entropy, mutual_information, variation_ratio]))
