import json
from pathlib import Path

file_path = 'scripts/calibrate_baseline_v2.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

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

content = content.replace('def calibrate(checkpoint_path: str, predictions_path: str):', custom_encoder + '\ndef calibrate(checkpoint_path: str, predictions_path: str):')
content = content.replace('json.dump({', 'json.dump({') # just to find it
content = content.replace('indent=4\n    )', 'indent=4, cls=NumpyEncoder\n    )')

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)
