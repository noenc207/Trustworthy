import json
from pprint import pprint
from src.core.config import get_config
from src.modules.evaluation.calibration.expected_calibration_error import expected_calibration_error
from src.modules.reporting.publication_tables import create_metrics_table
from src.modules.reporting.markdown_generator import dataframe_to_markdown
from src.modules.research.governance.claim_consistency import ClaimConsistencyValidator
from src.modules.research.statistics.test_selector import TestSelector

def show_parameters():
    print("="*60)
    print("1. TOÀN BỘ THÔNG SỐ CẤU HÌNH HỆ THỐNG (SYSTEM CONFIGURATION)")
    print("="*60)
    settings = get_config()
    # Pydantic v2 model_dump
    config_dict = settings.model_dump()
    print(json.dumps(config_dict, indent=4, default=str))
    
    print("\n" + "="*60)
    print("2. CHẠY THỬ NGHIỆM TÍNH TOÁN METRIC (EVALUATION LAYER)")
    print("="*60)
    import numpy as np
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.3, 0.4, 0.7])
    ece = expected_calibration_error(y_true, y_prob, num_bins=3)
    print(f"[Calibration] Expected Calibration Error (ECE): {ece:.4f}")
    
    print("\n" + "="*60)
    print("3. KIỂM ĐỊNH TỰ ĐỘNG VÀ QUẢN TRỊ KHOA HỌC (GOVERNANCE LAYER)")
    print("="*60)
    # Test Selector
    selector = TestSelector()
    data1 = np.random.normal(0, 1, 100).tolist()
    data2 = np.random.normal(0.5, 1, 100).tolist()
    test_type = selector.compare_two_groups(data1, data2, independent=False)
    print(f"[Test Selector] Khuyên dùng kiểm định: {test_type['test_name']} (p={test_type['p_value']:.4f})")
    
    # Claim Validator
    claim_val = ClaimConsistencyValidator()
    claim1 = claim_val.validate_claim("This model is SOTA", evidence={"benchmark": "ISIC_2019"})
    claim2 = claim_val.validate_claim("Results are statistically significant", evidence={"p_value": 0.02})
    claim3 = claim_val.validate_claim("Results are statistically significant", evidence={"p_value": 0.15})
    print(f"[Claim Validation] Yêu cầu 'SOTA' + Benchmark ISIC: {claim1}")
    print(f"[Claim Validation] Yêu cầu 'Significant' + p=0.02: {claim2}")
    print(f"[Claim Validation] Yêu cầu 'Significant' + p=0.15: {claim3}")
    
    print("\n" + "="*60)
    print("4. KẾT XUẤT BÁO CÁO (REPORTING LAYER)")
    print("="*60)
    metrics = {
        "ResNet-50 (Baseline)": {"Accuracy": 0.85, "AUC": 0.88, "ECE": 0.12},
        "Trustworthy AI (Ours)": {"Accuracy": 0.92, "AUC": 0.95, "ECE": 0.04},
    }
    df = create_metrics_table(metrics)
    md = dataframe_to_markdown(df)
    print(md)

if __name__ == "__main__":
    show_parameters()
