import os
import re

patterns = [
    (r'^\s*pass\s*$', 'PASS_STUB'),
    (r'raise NotImplementedError', 'NOT_IMPLEMENTED'),
    (r'#\s*TODO', 'TODO'),
    (r'#\s*FIXME', 'FIXME'),
    (r'#\s*HACK', 'HACK'),
    (r'placeholder', 'PLACEHOLDER'),
]

skip_dirs = ['.git', '__pycache__', 'node_modules', '.gemini']
results = []

for root, dirs, files in os.walk('src'):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if not f.endswith('.py'):
            continue
        filepath = os.path.join(root, f)
        try:
            with open(filepath, 'r', encoding='utf-8') as fh:
                lines = fh.readlines()
                for i, line in enumerate(lines):
                    for pat, kind in patterns:
                        if re.search(pat, line, re.IGNORECASE):
                            stripped = line.strip()
                            if kind == 'PASS_STUB':
                                if i > 0:
                                    prev = lines[i-1].strip()
                                    if 'Error' in prev or 'Exception' in prev:
                                        continue
                                    if 'class ' in prev and ':' in prev:
                                        continue
                            if kind == 'PLACEHOLDER':
                                lower = stripped.lower()
                                if 'placeholder' in lower and ('comment' in lower or 'docstring' in lower):
                                    continue
                            results.append(f'{kind}|{filepath}:{i+1}|{stripped}')
        except Exception:
            pass

for r in results:
    print(r)
print(f'---TOTAL: {len(results)}---')
