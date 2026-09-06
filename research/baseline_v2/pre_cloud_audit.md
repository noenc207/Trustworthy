# Pre-Cloud Audit Report

## Phase 0: Repository Freeze
- **Git tag**: `pre-cloud-audit` created
- **Commit**: `2810a9a` — "pre-cloud-audit: snapshot before comprehensive integrity audit"
- **Branch**: `main`

---

## Phase 1: Full Static Audit

### Category A (Harmless / Test / Documentation)
| FILE | LINE | CODE | REASON |
|------|------|------|--------|
| `tests/test_core.py` | various | `fake` | Test fixtures only |
| `tests/test_research_integrity.py` | various | `fake` | Test validation |
| `src/framework/testing/mocks.py` | all | `MockClassifier` etc. | Framework test infrastructure, not research |
| `src/deployment/model_export.py` | 42, 70, 100, 122 | `dummy_input` | Standard PyTorch ONNX/JIT tracing practice |
| `src/training/metrics.py` | 72-81 | `np.random.seed`, `np.random.randint` | Bootstrap CI computation — legitimate |
| `src/training/augmentation.py` | 127-163 | `np.random.beta`, `np.random.randint` | CutMix/MixUp augmentation — legitimate |
| `src/training/sampling.py` | 51-94 | `np.random.shuffle`, `np.random.choice` | Class-balanced sampling — legitimate |
| `src/evaluation/evaluate_pipeline.py` | 122 | `tpr >= 0.95` | Clinical threshold — legitimate |

### Category B (Real Bugs — FIXED)
| FILE | LINE | PROBLEM | FIX |
|------|------|---------|-----|
| `src/training/train_pipeline.py` | 37-38 | `except Exception: pass` — bare exception swallows errors | Changed to `except ImportError: logger.warning(...)` |
| `src/api/db/repositories/base.py` | - | PEP 695 generics incompatible with Python 3.11 | Fixed to `typing.Generic[T]` |
| `src/framework/common/registry.py` | - | PEP 695 generics incompatible with Python 3.11 | Fixed to `typing.Generic[T]` |

### Category C (Research Integrity — FIXED)
| FILE | LINE | PROBLEM | FIX |
|------|------|---------|-----|
| `scripts/evaluate_ood.py` | 62 | `"mocked out until real dataset config"` | Replaced with real OOD inference pipeline |
| `src/training/data_module.py` | 44-46 | Synthetic dataset fallback | `raise ValueError("Synthetic mock dataset is forbidden")` |
| `src/training/train_pipeline.py` | 233-238 | Model fallback with `nn.Sequential` | `raise ValueError("Model configuration is missing")` |
| `scripts/train_baseline_v2.py` | 16-20 | Auto-override `data/test_dataset_v2` silently | Explicit `raise ValueError` for synthetic |

### Category E (Intentional / Legitimate)
| FILE | LINE | CODE | REASON |
|------|------|------|--------|
| `src/modules/ood/detector.py` | 58, 65 | "Conservative fallback" | Designed OOD rejection behavior |
| `src/modules/dataset_manager/config.py` | 33 | `base_path = "data/ham10000"` | Default config value |
| Various `except ImportError: pass` | - | Optional dependency handling | Correct pattern |

---

## Phase 2: Python Syntax / Import Integrity
- `python -m compileall src scripts tests` → **PASS** (after PEP 695 fix)
- `import src` → **PASS**
- `from src.training.train_pipeline import train` → **PASS**
- `from src.training.data_module import SkinLesionDataModule` → **PASS**
- `from src.modules.classification.classifier import SkinLesionClassifier` → **PASS**

## Phase 3: Dependency Integrity
| Package | Version |
|---------|---------|
| Python | 3.11.9 |
| PyTorch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| timm | 1.0.29 |
| albumentations | 1.3.1 |
| pytorch_lightning | 2.6.5 |
| numpy | 2.4.6 |
| scikit-learn | 1.9.0 |

- `pip check` → **No broken requirements found**
- `requirements-lock.txt` → Created (151 packages)

## Phase 4: Hydra Config Integrity
- Config resolves without `???` values
- Model, dataset, optimizer, scheduler, seed all present
- **PASS**

## Phase 5: Dataset Contract
- 10/10 dataset contract tests passed
- Dataset returns `[3, 224, 224]` tensors with valid labels

---

## Known Remaining Items

### OOD Evaluation
- **STATUS**: `NOT_AVAILABLE` — No OOD dataset currently configured
- This is correct behavior per Phase 21/25 specification
- Will produce real results when OOD dataset (e.g., Fitzpatrick17k) is added

### API Engine (`src/api/dependencies/engine.py:67`)
- Uses untrained weights for integration testing
- **Not on critical path** for research pipeline
- Acceptable for demo/API purposes only

### `strict=False` in `zip()` calls
- `src/modules/inference_engine/executor.py:225`
- `src/modules/inference_engine/engine.py:154`
- These are in the inference engine, not the training/evaluation pipeline
- Low priority but should be changed to `strict=True` before production
