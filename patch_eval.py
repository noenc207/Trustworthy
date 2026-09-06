import json
from pathlib import Path

# Fix the JSON serialization issue in evaluate_baseline_v2.py
file_path = 'scripts/evaluate_baseline_v2.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace compute_metrics return to convert numpy types to native python types
# We will just write a custom json encoder
custom_encoder = '''
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
'''

content = content.replace('def evaluate(checkpoint_path: str, split: str = "test"):', custom_encoder + '\ndef evaluate(checkpoint_path: str, split: str = "test"):')
content = content.replace('json.dump(metrics, f, indent=4)', 'json.dump(metrics, f, indent=4, cls=NumpyEncoder)')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
