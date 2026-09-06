# Research Integrity Audit

| FILE | LINE | PROBLEM | ACTION TAKEN | STATUS |
|---|---|---|---|---|
| scripts/evaluate_baseline_v2.py | 55-63 | Hard-coded cfg = OmegaConf.create(...) with data/test_dataset_v2 | Removed hard-coded config. Extract config directly from checkpoint model_module.hparams or CLI. | FIXED |
| src/training/data_module.py | 43-45 | if dataset_name == "synthetic": self._setup_synthetic(stage) | Removed _setup_synthetic mock path completely. Raised error if dataset is synthetic. | FIXED |
| src/training/train_pipeline.py | 286-291 | Model fallback if cfg.model missing | Removed fallback logic. If cfg.model is missing, raise ValueError. | FIXED |
| scripts/train_baseline_v2.py | 16-20 | dataset.name == "synthetic" forced to data/test_dataset_v2 | Removed fallback override. Only use CLI provided args. | FIXED |
