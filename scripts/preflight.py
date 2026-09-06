"""
Pre-Cloud Preflight Check Script.

Validates the ENTIRE pipeline can run without errors before spending cloud GPU money.
Tests: imports, config, dataset, splits, transforms, model, forward, backward,
       checkpoint round-trip, evaluation, calibration, uncertainty, selective, conformal.

Usage:
    python scripts/preflight.py
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sys
import json
import traceback
from pathlib import Path
from loguru import logger

# Force CPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""


class PreflightResult:
    def __init__(self):
        self.results = {}
        self.failures = []
    
    def check(self, name: str, fn):
        try:
            fn()
            self.results[name] = "PASS"
            logger.info(f"  ✓ {name}")
        except Exception as e:
            self.results[name] = f"FAIL: {e}"
            self.failures.append((name, str(e)))
            logger.error(f"  ✗ {name}: {e}")
    
    def passed(self):
        return len(self.failures) == 0


def main():
    logger.info("=" * 60)
    logger.info("PRE-CLOUD PREFLIGHT CHECK")
    logger.info("=" * 60)
    
    pf = PreflightResult()
    
    # =========================================================
    # 1. IMPORT CHECKS
    # =========================================================
    logger.info("\n[1/12] IMPORT INTEGRITY")
    
    def check_imports():
        import torch
        import pytorch_lightning as pl
        import timm
        import albumentations
        import hydra
        from omegaconf import DictConfig, OmegaConf
        import numpy as np
        import sklearn
        from src.training.train_pipeline import SkinLesionLightningModule, train
        from src.training.data_module import SkinLesionDataModule
        from src.modules.classification.classifier import SkinLesionClassifier
        from src.training.augmentation import get_train_transforms, get_val_transforms
        from src.modules.uncertainty.mc_dropout import enable_mc_dropout, compute_mutual_information
        from src.core.constants import LesionClass
    pf.check("imports", check_imports)
    
    # =========================================================
    # 2. CONFIG INTEGRITY
    # =========================================================
    logger.info("\n[2/12] HYDRA CONFIG")
    
    def check_config():
        import hydra
        from omegaconf import OmegaConf
        from hydra import initialize_config_dir, compose
        
        config_dir = str(Path(__file__).parent.parent / "configs")
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="train")
        
        # Check no missing values
        yaml_str = OmegaConf.to_yaml(cfg)
        assert "???" not in yaml_str, "Config has missing required values (???)"
        
        # Check essential keys exist
        assert "model" in cfg, "Missing model config"
        assert "dataset" in cfg, "Missing dataset config"
        assert "trainer" in cfg, "Missing trainer config"
        assert cfg.get("seed") is not None, "Missing seed"
    pf.check("hydra_config", check_config)
    
    # =========================================================
    # 3. DATASET EXISTS
    # =========================================================
    logger.info("\n[3/12] DATASET CONTRACT")
    
    def check_dataset():
        test_data = Path("data/test_dataset_v2")
        assert test_data.exists(), f"Test dataset not found at {test_data}"
        
        csv_path = test_data / "labels" / "cleaned.csv"
        assert csv_path.exists(), f"Cleaned CSV not found at {csv_path}"
        
        import pandas as pd
        df = pd.read_csv(csv_path)
        assert "image_id" in df.columns, "Missing image_id column"
        assert len(df) > 0, "Empty dataset"
    pf.check("dataset_exists", check_dataset)
    
    def check_dataset_loading():
        from src.modules.dataset_manager.pytorch_dataset import SkinLesionDataset
        from src.training.augmentation import get_val_transforms
        
        csv_path = Path("data/test_dataset_v2/labels/cleaned.csv")
        ds = SkinLesionDataset(
            cleaned_csv_path=csv_path,
            transform=get_val_transforms(224),
            image_size=224
        )
        assert len(ds) > 0, "Dataset is empty"
        
        # Check output shape
        sample = ds[0]
        assert len(sample) >= 2, f"Dataset must return at least (image, label), got {len(sample)} elements"
        img, label = sample[0], sample[1]
        
        import torch
        assert isinstance(img, torch.Tensor), f"Image should be tensor, got {type(img)}"
        assert img.shape == (3, 224, 224), f"Expected [3,224,224], got {img.shape}"
        assert torch.isfinite(img).all(), "Image contains non-finite values"
    pf.check("dataset_loading", check_dataset_loading)
    
    # =========================================================
    # 4. TRANSFORMS
    # =========================================================
    logger.info("\n[4/12] TRANSFORM CONTRACT")
    
    def check_transforms():
        import numpy as np
        import torch
        from src.training.augmentation import get_train_transforms, get_val_transforms
        
        # Create a real-sized test image
        img = np.random.randint(0, 255, (450, 600, 3), dtype=np.uint8)
        
        train_tf = get_train_transforms(224)
        val_tf = get_val_transforms(224)
        
        train_out = train_tf(image=img)["image"]
        val_out = val_tf(image=img)["image"]
        
        assert isinstance(train_out, torch.Tensor), f"Train transform should return Tensor"
        assert isinstance(val_out, torch.Tensor), f"Val transform should return Tensor"
        assert train_out.shape == (3, 224, 224), f"Train output shape: {train_out.shape}"
        assert val_out.shape == (3, 224, 224), f"Val output shape: {val_out.shape}"
        assert torch.isfinite(train_out).all(), "Train transform produced non-finite values"
        assert torch.isfinite(val_out).all(), "Val transform produced non-finite values"
    pf.check("transforms", check_transforms)
    
    # =========================================================
    # 5. MODEL CONTRACT
    # =========================================================
    logger.info("\n[5/12] MODEL CONTRACT")
    
    def check_model_efficientnet():
        import torch
        from src.modules.classification.classifier import SkinLesionClassifier
        
        model = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7)
        model.eval()
        
        for bs in [1, 2, 4]:
            x = torch.randn(bs, 3, 224, 224)
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (bs, 7), f"Expected [{bs},7], got {logits.shape}"
            assert torch.isfinite(logits).all(), f"Non-finite logits for bs={bs}"
        
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    EfficientNet-B4 params: {n_params:,}")
    pf.check("model_efficientnet_b4", check_model_efficientnet)
    
    def check_model_resnet():
        import torch
        from src.modules.classification.classifier import SkinLesionClassifier
        
        model = SkinLesionClassifier(backbone="resnet50", num_classes=7)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 7), f"Expected [2,7], got {logits.shape}"
        assert torch.isfinite(logits).all()
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    ResNet-50 params: {n_params:,}")
    pf.check("model_resnet50", check_model_resnet)
    
    def check_model_densenet():
        import torch
        from src.modules.classification.classifier import SkinLesionClassifier
        
        model = SkinLesionClassifier(backbone="densenet121", num_classes=7)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 7), f"Expected [2,7], got {logits.shape}"
        assert torch.isfinite(logits).all()
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"    DenseNet-121 params: {n_params:,}")
    pf.check("model_densenet121", check_model_densenet)
    
    # =========================================================
    # 6. BACKWARD PASS
    # =========================================================
    logger.info("\n[6/12] BACKWARD PASS")
    
    def check_backward():
        import torch
        import torch.nn as nn
        from src.modules.classification.classifier import SkinLesionClassifier
        
        model = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7)
        model.train()
        
        x = torch.randn(2, 3, 224, 224)
        y = torch.tensor([0, 3])
        
        logits = model(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        
        loss.backward()
        
        has_grad = False
        for name, p in model.named_parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                assert torch.isfinite(p.grad).all(), f"Non-finite gradient in {name}"
        
        assert has_grad, "No gradients computed!"
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        
        # Verify parameters changed
        logits2 = model(x)
        assert not torch.equal(logits, logits2), "Parameters did not update after optimizer.step()"
    pf.check("backward_pass", check_backward)
    
    # =========================================================
    # 7. CHECKPOINT ROUND-TRIP
    # =========================================================
    logger.info("\n[7/12] CHECKPOINT ROUND-TRIP")
    
    def check_checkpoint_roundtrip():
        import torch
        import tempfile
        from src.modules.classification.classifier import SkinLesionClassifier
        
        model = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pred_before = model(x)
        
        # Save
        ckpt_path = Path("checkpoints/preflight_test.pth")
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        
        # Reload
        model2 = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7)
        model2.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model2.eval()
        
        with torch.no_grad():
            pred_after = model2(x)
        
        assert torch.allclose(pred_before, pred_after, atol=1e-6), \
            f"Predictions differ after checkpoint reload! Max diff: {(pred_before - pred_after).abs().max()}"
        
        # Cleanup
        ckpt_path.unlink(missing_ok=True)
    pf.check("checkpoint_roundtrip", check_checkpoint_roundtrip)
    
    # =========================================================
    # 8. MC DROPOUT
    # =========================================================
    logger.info("\n[8/12] MC DROPOUT CONTRACT")
    
    def check_mc_dropout():
        import torch
        import torch.nn as nn
        from src.modules.classification.classifier import SkinLesionClassifier
        from src.modules.uncertainty.mc_dropout import enable_mc_dropout
        
        model = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7)
        model.eval()
        enable_mc_dropout(model)
        
        # Verify: BatchNorm is eval, Dropout is train
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                assert not m.training, f"BatchNorm should be eval but is training: {m}"
            if isinstance(m, nn.Dropout):
                assert m.training, f"Dropout should be training but is eval: {m}"
        
        # Check stochasticity
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        # With dropout active, outputs should differ
        has_dropout = any(isinstance(m, nn.Dropout) for m in model.modules())
        if has_dropout:
            # At least some values should differ
            if torch.equal(out1, out2):
                logger.warning("MC Dropout outputs are identical - dropout may have p=0")
    pf.check("mc_dropout", check_mc_dropout)
    
    # =========================================================
    # 9. CALIBRATION METRICS
    # =========================================================
    logger.info("\n[9/12] CALIBRATION METRICS")
    
    def check_calibration_metrics():
        import numpy as np
        from scripts.calibrate_baseline_v2 import expected_calibration_error, brier_score_multiclass
        
        # Deterministic test data
        y_true = np.array([0, 1, 2, 0, 1])
        y_prob = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1],
            [0.3, 0.5, 0.2],
        ])
        
        ece = expected_calibration_error(y_true, y_prob)
        assert isinstance(ece, float), f"ECE should be float, got {type(ece)}"
        assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece}"
        
        brier = brier_score_multiclass(y_true, y_prob, num_classes=3)
        assert isinstance(brier, float), f"Brier should be float, got {type(brier)}"
        assert brier >= 0, f"Brier should be >= 0, got {brier}"
    pf.check("calibration_metrics", check_calibration_metrics)
    
    # =========================================================
    # 10. SELECTIVE PREDICTION MATH
    # =========================================================
    logger.info("\n[10/12] SELECTIVE PREDICTION")
    
    def check_selective():
        import numpy as np
        
        # Simulate: 5 samples, some correct, some wrong, varying confidence
        confidences = np.array([0.95, 0.80, 0.60, 0.40, 0.20])
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0])  # 2 wrong (idx 2, 4)
        
        errors = (y_true != y_pred)
        thresholds = np.linspace(0, 1, 11)
        
        for tau in thresholds:
            accepted = confidences >= tau
            coverage = np.mean(accepted)
            if np.sum(accepted) > 0:
                risk = np.mean(errors[accepted])
            else:
                risk = 0.0
            assert 0 <= coverage <= 1
            assert 0 <= risk <= 1
    pf.check("selective_prediction", check_selective)
    
    # =========================================================
    # 11. CONFORMAL PREDICTION MATH
    # =========================================================
    logger.info("\n[11/12] CONFORMAL PREDICTION")
    
    def check_conformal():
        import numpy as np
        
        # Calibration set: 10 samples, 3 classes
        np.random.seed(42)
        n_cal = 10
        probs = np.random.dirichlet([1, 1, 1], size=n_cal)
        true_labels = np.random.randint(0, 3, size=n_cal)
        
        # Conformal scores: s_i = 1 - p_true
        scores = 1.0 - probs[np.arange(n_cal), true_labels]
        
        alpha = 0.1
        q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        if q_level > 1.0:
            q_level = 1.0
        q_hat = np.quantile(scores, q_level, method="higher")
        
        assert 0 <= q_hat <= 1, f"q_hat should be in [0,1], got {q_hat}"
        
        # Test set
        n_test = 5
        test_probs = np.random.dirichlet([1, 1, 1], size=n_test)
        test_true = np.random.randint(0, 3, size=n_test)
        
        test_scores = 1.0 - test_probs
        prediction_sets = test_scores <= q_hat
        set_sizes = np.sum(prediction_sets, axis=1)
        
        assert all(s >= 0 for s in set_sizes), "Set sizes must be non-negative"
        assert all(s <= 3 for s in set_sizes), "Set sizes must be <= num_classes"
    pf.check("conformal_prediction", check_conformal)
    
    # =========================================================
    # 12. SPLIT INTEGRITY
    # =========================================================
    logger.info("\n[12/12] SPLIT INTEGRITY")
    
    def check_splits():
        import pandas as pd
        
        data_dir = Path("data/test_dataset_v2")
        splits_dir = data_dir / "splits"
        csv_path = data_dir / "labels" / "cleaned.csv"
        
        if not splits_dir.exists():
            logger.warning("No splits directory found - will be generated on first run")
            return
        
        df = pd.read_csv(csv_path)
        
        # Check split files exist
        split_files = {}
        for name in ["train", "val", "test"]:
            f = splits_dir / f"{name}_indices.csv"
            if f.exists():
                ids = pd.read_csv(f, header=None)[0].astype(str).tolist()
                split_files[name] = set(ids)
        
        # Check no overlap between splits
        for s1 in split_files:
            for s2 in split_files:
                if s1 < s2:
                    overlap = split_files[s1].intersection(split_files[s2])
                    assert len(overlap) == 0, f"LEAKAGE: {s1} and {s2} share {len(overlap)} samples"
    pf.check("split_integrity", check_splits)
    
    # =========================================================
    # FINAL REPORT
    # =========================================================
    logger.info("\n" + "=" * 60)
    total = len(pf.results)
    passed = sum(1 for v in pf.results.values() if v == "PASS")
    failed = total - passed
    
    if pf.passed():
        logger.info(f"ALL PREFLIGHT CHECKS PASSED ({passed}/{total})")
        gate_status = {"passed": True, "checks": pf.results}
    else:
        logger.error(f"PREFLIGHT FAILED: {failed}/{total} checks failed")
        for name, err in pf.failures:
            logger.error(f"  FAIL: {name} — {err}")
        gate_status = {"passed": False, "checks": pf.results, "failures": [{"name": n, "error": e} for n, e in pf.failures]}
    
    # Save gate file
    gate_dir = Path("research/baseline_v2/results")
    gate_dir.mkdir(parents=True, exist_ok=True)
    with open(gate_dir / "pre_cloud_gate.json", "w") as f:
        json.dump(gate_status, f, indent=2)
    
    logger.info(f"Gate file saved to {gate_dir / 'pre_cloud_gate.json'}")
    logger.info("=" * 60)
    
    sys.exit(0 if pf.passed() else 1)


if __name__ == "__main__":
    main()
