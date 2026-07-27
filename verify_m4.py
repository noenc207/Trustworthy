"""
Verification script for Milestone 4 — Repository Layer.
Run from project root: python verify_m4.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

errors = []
passed = []

def ok(msg):
    passed.append(msg)
    print(f"  [PASS] {msg}")

def fail(msg, err):
    errors.append(f"{msg}: {err}")
    print(f"  [FAIL] {msg}: {err}")

print("\n=== Milestone 4 Verification: Repository Layer ===\n")

# 1. BaseRepository generics and methods
try:
    import inspect

    from src.api.db.repositories.base import BaseRepository
    methods = [m[0] for m in inspect.getmembers(BaseRepository, predicate=inspect.iscoroutinefunction)]
    assert "get" in methods
    assert "get_multi" in methods
    assert "create" in methods
    assert "update" in methods
    assert "delete" in methods
    ok("repositories.base — BaseRepository has all async CRUD methods")
except Exception as e:
    fail("repositories.base", e)

# 2. UserRepository
try:
    from unittest.mock import Mock

    from src.api.db.models.user import User
    from src.api.db.repositories import UserRepository

    mock_session = Mock()
    repo = UserRepository(session=mock_session)
    assert repo.model == User
    assert hasattr(repo, "get_by_email")
    assert hasattr(repo, "get_by_username")
    ok("repositories.user — UserRepository initialized and extends base methods")
except Exception as e:
    fail("repositories.user", e)

# 3. UploadRepository
try:
    from unittest.mock import Mock

    from src.api.db.models.upload import Upload
    from src.api.db.repositories import UploadRepository

    mock_session = Mock()
    repo = UploadRepository(session=mock_session)
    assert repo.model == Upload
    assert hasattr(repo, "get_by_user")
    ok("repositories.upload — UploadRepository initialized and extends base methods")
except Exception as e:
    fail("repositories.upload", e)

# 4. PredictionRepository
try:
    from unittest.mock import Mock

    from src.api.db.models.prediction import Prediction
    from src.api.db.repositories import PredictionRepository

    mock_session = Mock()
    repo = PredictionRepository(session=mock_session)
    assert repo.model == Prediction
    assert hasattr(repo, "get_by_upload")
    assert hasattr(repo, "get_by_user")
    ok("repositories.prediction — PredictionRepository initialized and extends base methods")
except Exception as e:
    fail("repositories.prediction", e)

# Summary
print(f"\n=== Results: {len(passed)} passed, {len(errors)} failed ===\n")
if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print("  All checks passed. Milestone 4 complete.")
    sys.exit(0)
