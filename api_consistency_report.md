# API Consistency Report

All internal Domain Transfer Objects (DTOs) match:
- `@dataclass(frozen=True)`
- Common fields: `valid: bool`, `status: str`, `runtime_ms: float`, `warnings: list[str]`.
- Enforced Typing via standard library `typing`.
- Consistent Error Handling: `ClassifierDomainError` subclasses trap issues cleanly without stack trace leakage to the HTTP layer.
