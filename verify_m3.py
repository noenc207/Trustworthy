"""
Verification script for Milestone 3 — Security Layer.
Run from project root: python verify_m3.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

errors = []
passed = []

def ok(msg):
    passed.append(msg)
    print(f"  [PASS] {msg}")

def fail(msg, err):
    errors.append(f"{msg}: {err}")
    print(f"  [FAIL] {msg}: {err}")

print("\n=== Milestone 3 Verification: Security Layer ===\n")

# 1. Password hashing
try:
    from src.api.security import get_password_hash, verify_password
    plain = "SuperSecret123!"
    hashed = get_password_hash(plain)
    assert hashed != plain, "Hash matches plain text!"
    assert verify_password(plain, hashed), "Valid password rejected"
    assert not verify_password("WrongPassword!", hashed), "Invalid password accepted"
    ok("security.password — bcrypt hashing and verification working")
except Exception as e:
    fail("security.password", e)

# 2. JWT token generation & validation
try:
    from uuid import uuid4
    from src.api.security import create_access_token, create_refresh_token, verify_token
    from src.api.schemas.auth import TokenData
    
    subject_id = uuid4()
    access_token = create_access_token(subject=subject_id, email="test@example.com", scopes=["user"])
    refresh_token = create_refresh_token(subject=subject_id)
    
    token_data = verify_token(access_token, expected_type="access")
    assert isinstance(token_data, TokenData)
    assert token_data.sub == str(subject_id)
    assert token_data.email == "test@example.com"
    assert "user" in token_data.scopes
    
    refresh_data = verify_token(refresh_token, expected_type="refresh")
    assert refresh_data.sub == str(subject_id)
    
    ok("security.jwt — RS/HS256 access and refresh token lifecycle working")
except Exception as e:
    fail("security.jwt", e)

# 3. API Key verification
try:
    import asyncio
    from src.api.security import verify_api_key
    from src.core.exceptions import AuthException
    
    # Valid
    valid_key = "a_very_long_secure_api_key_123"
    result = asyncio.run(verify_api_key(valid_key))
    assert result == valid_key
    
    # Invalid
    try:
        asyncio.run(verify_api_key("short"))
        fail("security.api_keys", "Failed to reject short key")
    except AuthException:
        pass
        
    ok("security.api_keys — basic API key validation working")
except Exception as e:
    fail("security.api_keys", e)

# 4. Auth dependency wiring
try:
    from src.api.dependencies.auth import oauth2_scheme, get_current_user, get_current_token_payload
    import inspect
    assert oauth2_scheme.model.flows.password.tokenUrl == "/api/v1/auth/login"
    assert inspect.iscoroutinefunction(get_current_user)
    assert inspect.iscoroutinefunction(get_current_token_payload)
    ok("dependencies.auth — OAuth2PasswordBearer configured with correct tokenUrl")
except Exception as e:
    fail("dependencies.auth", e)

# Summary
print(f"\n=== Results: {len(passed)} passed, {len(errors)} failed ===\n")
if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print("  All checks passed. Milestone 3 complete.")
    sys.exit(0)
