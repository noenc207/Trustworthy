import re


def replace_in_file(filepath, pattern, replacement):
    with open(filepath) as f:
        content = f.read()
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    with open(filepath, 'w') as f:
        f.write(content)

replace_in_file('src/modules/inference_engine/engine.py', r'CLASS_LABELS = \[', r'CLASS_LABELS: typing.ClassVar[list[str]] = [')
replace_in_file('src/modules/inference_engine/engine.py', r'zip\(self\.CLASS_LABELS, probs_np\)', r'zip(self.CLASS_LABELS, probs_np, strict=False)')
replace_in_file('src/modules/inference_engine/events.py', r'class EventType\(str, Enum\):', r'class EventType(str, Enum):')
replace_in_file('src/modules/inference_engine/events.py', r'for priority, handler in handlers:', r'for _priority, handler in handlers:')
replace_in_file('src/modules/inference_engine/executor.py', r'raise TimeoutError\(', r'raise TimeoutError(')
replace_in_file('src/modules/inference_engine/executor.py', r'zip\(c_res\.class_labels, c_res\.probabilities\)', r'zip(c_res.class_labels, c_res.probabilities, strict=False)')
replace_in_file('src/modules/inference_engine/factory.py', r'class StageSlot\(str, Enum\):', r'class StageSlot(str, Enum):')
replace_in_file('src/modules/inference_engine/graph.py', r'any dependency \n', r'any dependency\n')
replace_in_file('src/modules/inference_engine/runtime_state.py', r'class RuntimeState\(str, Enum\):', r'class RuntimeState(str, Enum):')
replace_in_file('src/modules/preprocessing/operations/clahe.py', r'l, a, b = cv2.split\(lab\)', r'l_channel, a, b = cv2.split(lab)')
replace_in_file('src/modules/preprocessing/operations/clahe.py', r'cl = clahe.apply\(l\)', r'cl = clahe.apply(l_channel)')
replace_in_file('src/modules/preprocessing/visualization_enhanced.py', r'fig, axes = plt.subplots', r'_fig, axes = plt.subplots')

