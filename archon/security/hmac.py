import hmac
import hashlib
import os
import time
from typing import Optional

# Default clock skew tolerance (in seconds)
DEFAULT_TOLERANCE = 300  # 5 minutes


def _get_secret() -> str:
    secret = os.getenv("HMAC_SECRET", "")
    if not secret:
        # Intentionally return empty secret; caller should decide whether to enforce
        return ""
    return secret


def sign(body: bytes, timestamp: str, secret: Optional[str] = None) -> str:
    """Generate an HMAC SHA256 signature over timestamp + '.' + body.

    The canonical string is: f"{timestamp}." + body (raw bytes)
    Returns a hex digest string.
    """
    key = (secret if secret is not None else _get_secret()).encode("utf-8")
    msg = timestamp.encode("utf-8") + b"." + (body or b"")
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def verify(authorization_header: Optional[str], timestamp: Optional[str], body: bytes, 
           tolerance_seconds: int = DEFAULT_TOLERANCE, required: bool = True, secret: Optional[str] = None) -> (bool, str):
    """Verify HMAC Authorization header of form: 'HMAC <hex_signature>'.

    - authorization_header: value of 'Authorization' header
    - timestamp: value of 'X-Timestamp' header (epoch seconds or ISO8601 numeric seconds)
    - body: raw request body bytes
    - tolerance_seconds: allowed clock skew
    - required: if False, allow missing signature (returns True)
    - secret: optional explicit secret; else uses env HMAC_SECRET
    """
    if not required:
        return True, "HMAC not required"

    if not authorization_header or not authorization_header.strip().lower().startswith("hmac "):
        return False, "Missing or invalid Authorization header"

    if not timestamp:
        return False, "Missing X-Timestamp header"

    # Parse timestamp as float seconds
    try:
        ts = float(timestamp)
    except Exception:
        return False, "Invalid X-Timestamp format"

    now = time.time()
    if abs(now - ts) > float(tolerance_seconds):
        return False, "Timestamp outside allowed tolerance"

    provided_sig = authorization_header.split(" ", 1)[1].strip()
    expected_sig = sign(body or b"", str(int(ts)), secret=secret)

    if not hmac.compare_digest(provided_sig, expected_sig):
        return False, "Signature mismatch"

    return True, "OK"
