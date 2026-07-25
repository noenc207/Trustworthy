"""
AI Module Registry — Plugin System.

Central registry for all AI module implementations.
Researchers register new algorithms here; the engine picks them up via config.

Usage:
    # Register a new OOD detector:
    registry = AIModuleRegistry.instance()
    registry.register_ood_detector("vim", VIMDetector(model))

    # The engine retrieves it:
    detector = registry.get_ood_detector("vim")
"""
from __future__ import annotations

from typing import Any

from loguru import logger


class AIModuleRegistry:
    """
    Centralized plugin registry for all AI module categories.
    Implements the Singleton pattern — one shared instance across the app.
    """

    _instance: AIModuleRegistry | None = None

    def __init__(self) -> None:
        self._classifiers: dict[str, Any] = {}
        self._ood_detectors: dict[str, Any] = {}
        self._uncertainty_estimators: dict[str, Any] = {}
        self._calibrators: dict[str, Any] = {}
        self._explainers: dict[str, Any] = {}
        self._quality_assessors: dict[str, Any] = {}
        self._recommendation_engines: dict[str, Any] = {}
        self._preprocessors: dict[str, Any] = {}

    @classmethod
    def instance(cls) -> AIModuleRegistry:
        """Return the global registry singleton."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Generic internal helpers ───────────────────────────────────────────

    def _register(self, store: dict, name: str, module: Any) -> None:
        if name in store:
            logger.warning(f"Module '{name}' already registered in category {store}. Overwriting.")
        store[name] = module

    def _get(self, store: dict, name: str, category: str) -> Any:
        if name not in store:
            available = list(store.keys())
            raise KeyError(
                f"Module '{name}' not found in {category} registry. "
                f"Available: {available}"
            )
        return store[name]

    def _list(self, store: dict) -> list[str]:
        return list(store.keys())

    # ── Classifier ────────────────────────────────────────────────────────

    def register_classifier(self, name: str, classifier: Any) -> None:
        self._register(self._classifiers, name, classifier)
        logger.info(f"Registered classifier: '{name}'")

    def get_classifier(self, name: str) -> Any:
        return self._get(self._classifiers, name, "classifier")

    def list_classifiers(self) -> list[str]:
        return self._list(self._classifiers)

    # ── OOD Detector ──────────────────────────────────────────────────────

    def register_ood_detector(self, name: str, detector: Any) -> None:
        self._register(self._ood_detectors, name, detector)
        logger.info(f"Registered OOD detector: '{name}'")

    def get_ood_detector(self, name: str) -> Any:
        return self._get(self._ood_detectors, name, "ood_detector")

    def list_ood_detectors(self) -> list[str]:
        return self._list(self._ood_detectors)

    # ── Uncertainty Estimator ─────────────────────────────────────────────

    def register_uncertainty_estimator(self, name: str, estimator: Any) -> None:
        self._register(self._uncertainty_estimators, name, estimator)
        logger.info(f"Registered uncertainty estimator: '{name}'")

    def get_uncertainty_estimator(self, name: str) -> Any:
        return self._get(self._uncertainty_estimators, name, "uncertainty_estimator")

    def list_uncertainty_estimators(self) -> list[str]:
        return self._list(self._uncertainty_estimators)

    # ── Calibrator ────────────────────────────────────────────────────────

    def register_calibrator(self, name: str, calibrator: Any) -> None:
        self._register(self._calibrators, name, calibrator)
        logger.info(f"Registered calibrator: '{name}'")

    def get_calibrator(self, name: str) -> Any:
        return self._get(self._calibrators, name, "calibrator")

    def list_calibrators(self) -> list[str]:
        return self._list(self._calibrators)

    # ── Explainer ─────────────────────────────────────────────────────────

    def register_explainer(self, name: str, explainer: Any) -> None:
        self._register(self._explainers, name, explainer)
        logger.info(f"Registered explainer: '{name}'")

    def get_explainer(self, name: str) -> Any:
        return self._get(self._explainers, name, "explainer")

    def list_explainers(self) -> list[str]:
        return self._list(self._explainers)

    # ── Quality Assessor ──────────────────────────────────────────────────

    def register_quality_assessor(self, name: str, assessor: Any) -> None:
        self._register(self._quality_assessors, name, assessor)
        logger.info(f"Registered quality assessor: '{name}'")

    def get_quality_assessor(self, name: str) -> Any:
        return self._get(self._quality_assessors, name, "quality_assessor")

    def list_quality_assessors(self) -> list[str]:
        return self._list(self._quality_assessors)

    # ── Recommendation Engine ─────────────────────────────────────────────

    def register_recommendation_engine(self, name: str, engine: Any) -> None:
        self._register(self._recommendation_engines, name, engine)
        logger.info(f"Registered recommendation engine: '{name}'")

    def get_recommendation_engine(self, name: str) -> Any:
        return self._get(self._recommendation_engines, name, "recommendation_engine")

    def list_recommendation_engines(self) -> list[str]:
        return self._list(self._recommendation_engines)

    # ── Preprocessor ──────────────────────────────────────────────────────

    def register_preprocessor(self, name: str, preprocessor: Any) -> None:
        self._register(self._preprocessors, name, preprocessor)
        logger.info(f"Registered preprocessor: '{name}'")

    def get_preprocessor(self, name: str) -> Any:
        return self._get(self._preprocessors, name, "preprocessor")

    def list_preprocessors(self) -> list[str]:
        return self._list(self._preprocessors)

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict[str, list[str]]:
        """Return a summary of all registered modules."""
        return {
            "classifiers": self.list_classifiers(),
            "ood_detectors": self.list_ood_detectors(),
            "uncertainty_estimators": self.list_uncertainty_estimators(),
            "calibrators": self.list_calibrators(),
            "explainers": self.list_explainers(),
            "quality_assessors": self.list_quality_assessors(),
            "recommendation_engines": self.list_recommendation_engines(),
            "preprocessors": self.list_preprocessors(),
        }
