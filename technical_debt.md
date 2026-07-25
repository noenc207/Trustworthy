# Technical Debt Report

**Status:** Minimal

1. **Test Coverage**: We rely heavily on integration scripts (`verify_batch*.py`). A pure `pytest` suite migration covering 100% branch logic would take additional time.
2. **Environment**: Windows `Access Denied` issues surfaced during `__pycache__` deletion. Docker containerization is highly recommended for deployment.
3. **Memory Profiling**: `memory_mb` in `InferenceSession` is currently statically mapped or tracked via crude proxy; needs native `torch.cuda.memory_allocated()` injection on deployment.
