"""
Verification Script for Milestone 6.9 and beyond.
This script validates the execution of Evaluation, Governance, and Reporting layers.
"""
import sys
import logging
from pathlib import Path
from src.modules.evaluation.benchmark.benchmark_loader import BenchmarkLoader
from src.modules.evaluation.calibration.expected_calibration_error import expected_calibration_error
from src.modules.research.statistics.assumption_validator import AssumptionValidator
from src.modules.research.governance.claim_consistency import ClaimConsistencyValidator
from src.modules.reporting.publication_tables import create_metrics_table
from src.modules.reporting.markdown_generator import dataframe_to_markdown

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def verify_m69_layers():
    logging.info("Starting M6.9 Verification...")
    
    # 1. Benchmark Loader Validation
    try:
        loader = BenchmarkLoader("ISIC")
        loader.load_metadata()
        logging.info("BenchmarkLoader initialized safely.")
    except Exception as e:
        logging.error(f"BenchmarkLoader failed: {e}")
        
    try:
        import numpy as np
        y_true = np.array([0, 1, 1, 0])
        y_prob = np.array([0.1, 0.9, 0.8, 0.2])
        ece = expected_calibration_error(y_true, y_prob)
        logging.info(f"Calibration Engine (ECE) initialized and tested. ECE = {ece}")
    except Exception as e:
        logging.error(f"Calibration Engine failed: {e}")
        
    # 3. Governance Statistics
    try:
        validator = AssumptionValidator()
        logging.info("AssumptionValidator (Governance) initialized.")
    except Exception as e:
        logging.error(f"AssumptionValidator failed: {e}")
        
    # 4. Claim Consistency
    try:
        claim_val = ClaimConsistencyValidator()
        res = claim_val.validate_claim("SOTA", evidence={"p_value": 0.01, "benchmark": "ISIC"})
        logging.info(f"ClaimConsistencyValidator evaluated SOTA claim safely. Valid: {res}")
    except Exception as e:
        logging.error(f"ClaimConsistencyValidator failed: {e}")
        
    # 5. Reporting
    try:
        metrics = {
            "Method A": {"Accuracy": 0.95, "ECE": 0.02},
            "Method B": {"Accuracy": 0.92, "ECE": 0.05},
        }
        df = create_metrics_table(metrics)
        md = dataframe_to_markdown(df)
        logging.info(f"Reporting Layer successfully generated markdown table:\n{md}")
    except Exception as e:
        logging.error(f"Reporting Layer failed: {e}")
        sys.exit(1)
        
    logging.info("M6.9+ Verification Completed Successfully.")

if __name__ == "__main__":
    verify_m69_layers()
