"""Final repository scan — filters out legitimate abstract/exception stubs."""
import os
import re

real_issues = []

for root, dirs, files in os.walk('src'):
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__']]
    for f in files:
        if not f.endswith('.py'):
            continue
        filepath = os.path.join(root, f)
        with open(filepath, 'r', encoding='utf-8') as fh:
            lines = fh.readlines()
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # --- pass stubs ---
                if stripped == 'pass':
                    if any(k in f for k in ['interfaces.py', 'base.py', 'exceptions.py']):
                        continue
                    is_legit = False
                    for j in range(max(0, i-3), i):
                        prev = lines[j].strip()
                        if any(k in prev for k in ['Error', 'Exception', '@abstractmethod']):
                            is_legit = True
                            break
                    if is_legit:
                        continue
                    real_issues.append(f'PASS|{filepath}:{i+1}')
                
                # --- placeholder text ---
                elif 'placeholder' in stripped.lower():
                    real_issues.append(f'PLACEHOLDER|{filepath}:{i+1}|{stripped}')
                
                # --- NotImplementedError ---
                elif re.search(r'raise NotImplementedError', stripped):
                    if any(k in f for k in ['interfaces.py', 'base.py', 'wrappers']):
                        continue
                    real_issues.append(f'NOT_IMPL|{filepath}:{i+1}|{stripped}')
                
                # --- TODO/FIXME/HACK ---
                elif re.search(r'#\s*(TODO|FIXME|HACK)', stripped, re.IGNORECASE):
                    real_issues.append(f'TODO_FIXME|{filepath}:{i+1}|{stripped}')

if real_issues:
    print(f'REMAINING REAL ISSUES: {len(real_issues)}')
    for r in real_issues:
        print(r)
else:
    print('ZERO REAL ISSUES REMAINING. REPOSITORY IS CLEAN.')
