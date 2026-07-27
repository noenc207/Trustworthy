"""
Verification script for Milestone 2 — ORM Models.
Run from project root: python verify_m2.py
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

print("\n=== Milestone 2 Verification: ORM Models ===\n")

# 1. Models load without error
try:
    from src.api.db.models import Prediction, Upload, User
    ok("models package — User, Upload, Prediction imported successfully")
except Exception as e:
    fail("models package", e)

# 2. Check User relationships
try:
    from sqlalchemy.orm import class_mapper
    mapper = class_mapper(User)
    assert "uploads" in mapper.relationships, "User missing 'uploads' relationship"
    assert "predictions" in mapper.relationships, "User missing 'predictions' relationship"
    ok("User model — relationships configured correctly")
except Exception as e:
    fail("User model", e)

# 3. Check Upload relationships
try:
    mapper = class_mapper(Upload)
    assert "user" in mapper.relationships, "Upload missing 'user' relationship"
    assert "prediction" in mapper.relationships, "Upload missing 'prediction' relationship"
    ok("Upload model — relationships configured correctly")
except Exception as e:
    fail("Upload model", e)

# 4. Check Prediction relationships & JSONB
try:
    mapper = class_mapper(Prediction)
    assert "user" in mapper.relationships, "Prediction missing 'user' relationship"
    assert "upload" in mapper.relationships, "Prediction missing 'upload' relationship"

    # Check JSONB
    col = Prediction.__table__.columns["full_result"]
    assert type(col.type).__name__ == "JSONB", f"Expected JSONB, got {type(col.type).__name__}"
    ok("Prediction model — relationships & JSONB configured correctly")
except Exception as e:
    fail("Prediction model", e)

# 5. Check Alembic migration script
try:
    import glob
    migrations = glob.glob("alembic/versions/*_create_initial_tables.py")
    assert len(migrations) == 1, "Missing create_initial_tables migration"
    with open(migrations[0]) as f:
        content = f.read()
    assert "op.create_table(\n        'users'" in content, "Missing users table in migration"
    assert "op.create_table(\n        'uploads'" in content, "Missing uploads table in migration"
    assert "op.create_table(\n        'predictions'" in content, "Missing predictions table in migration"
    ok("Alembic migration — create_initial_tables exists and contains tables")
except Exception as e:
    fail("Alembic migration", e)

# Summary
print(f"\n=== Results: {len(passed)} passed, {len(errors)} failed ===\n")
if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print("  All checks passed. Milestone 2 complete.")
    sys.exit(0)
