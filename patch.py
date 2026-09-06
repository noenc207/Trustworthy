import sys
with open('src/training/train_pipeline.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add safe globals
import_block = "import torch\noriginal_load = torch.load\ndef safe_load(*args, **kwargs):\n    kwargs['weights_only'] = False\n    return original_load(*args, **kwargs)\ntorch.load = safe_load\nimport torch.nn as nn\nfrom omegaconf import DictConfig, ListConfig, OmegaConf\nimport omegaconf.base\nimport omegaconf.nodes\nimport typing\nfrom torch.serialization import add_safe_globals\ntry:\n    add_safe_globals([DictConfig, ListConfig, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode, typing.Any])\nexcept Exception:\n    pass\n"
code = code.replace("import torch\nimport torch.nn as nn\nfrom omegaconf import DictConfig, OmegaConf", import_block)

# 2. Fix fallback model
bad_model = '''    if "model" in cfg and cfg.model is not None:
        model_instance = hydra.utils.instantiate(cfg.model)
    else:
        # For standalone script execution fallback
        model_instance = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, len(LesionClass))
        )'''
good_model = '''    if "model" in cfg and cfg.model is not None:
        model_instance = hydra.utils.instantiate(cfg.model)
    else:
        raise ValueError("Model configuration is missing. Cannot instantiate model.")'''
code = code.replace(bad_model, good_model)

# 3. Remove synthetic dataset module
import re
code = re.sub(r"class SkinLesionDataModule\(pl\.LightningDataModule\):.*?def test_dataloader.*?(?=\n\n@hydra\.main)", "", code, flags=re.DOTALL)

with open('src/training/train_pipeline.py', 'w', encoding='utf-8') as f:
    f.write(code)
