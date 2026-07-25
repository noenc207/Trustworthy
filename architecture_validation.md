# Architecture Validation

**Constraint Validation:**
- ✅ Classifier Domain depends ONLY on Torch/Torchvision. It does NOT depend on Explainability, Calibration, etc.
- ✅ Dependency direction STRICTLY maintained: Classifier -> Explainability -> Calibration.
- ✅ No circular imports detected.
- ✅ ModelFactory abstracts completely the instantiation logic.
