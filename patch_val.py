import sys
with open('scripts/validate_splits.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace('base_dir = Path("d:/Trustworthy/data/datasets")
    if not base_dir.exists():
        base_dir = Path("d:/Trustworthy/data")
    datasets = ["ham10000", "isic2019", "test_dataset", "test_dataset_v2"]', 
'''    base_dir = Path("d:/Trustworthy/data/datasets")
    if not base_dir.exists():
        base_dir = Path("d:/Trustworthy/data")
    datasets = ["ham10000", "isic2019", "test_dataset", "test_dataset_v2"]''')

with open('scripts/validate_splits.py', 'w', encoding='utf-8') as f:
    f.write(code)
