"""
Token verification for Neon Auth (Managed Better Auth).

Why this replaces the custom JWT_SECRET + Google OAuth code:
  - Neon Auth runs its own hosted auth service tied to your Neon project
    (the same project you already use for the LangGraph checkpointer).
  - It handles the Google sign-in flow itself (dev mode even ships shared
    Google credentials, so you don't need your own Google Cloud OAuth client
    while developing), and syncs signed-in users into your Postgres under
    the `neon_auth` schema automatically — no users table or upsert code
    for us to write or maintain.
  - It issues short-lived (15 min) EdDSA-signed JWTs. We verify those here
    against Neon's public JWKS endpoint — there is no shared secret for us
    to generate or store, which is the point: we never hold a signing key
    that could leak.

Required env var:
  NEON_AUTH_BASE_URL — from the Neon console: your project -> Settings -> Auth.
  Looks like: https://<something>.neon-auth.com  (or similar; copy it exactly
  as shown in the console, no trailing slash).
"""
import os
import base64
import time
from typing import Optional
from urllib.parse import urlparse

import jwt
import requests
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from jwt import PyJWTError

NEON_AUTH_BASE_URL = os.environ["NEON_AUTH_BASE_URL"].rstrip("/")
NEON_JWKS_URL = f"{NEON_AUTH_BASE_URL}/.well-known/jwks.json"

_parsed = urlparse(NEON_AUTH_BASE_URL)
ORIGIN = f"{_parsed.scheme}://{_parsed.netloc}"

_jwks_cache: dict | None = None
_jwks_fetched_at: float = 0.0
_JWKS_TTL_SECONDS = 600  # refetch at most every 10 minutes, or on an unknown kid


def _get_jwks(force: bool = False) -> dict:
    global _jwks_cache, _jwks_fetched_at
    if force or _jwks_cache is None or (time.time() - _jwks_fetched_at) > _JWKS_TTL_SECONDS:
        response = requests.get(NEON_JWKS_URL, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_fetched_at = time.time()
    return _jwks_cache


def _get_signing_key(token: str, jwks: dict) -> Ed25519PublicKey:
    kid = jwt.get_unverified_header(token)["kid"]
    for jwk in jwks["keys"]:
        if jwk["kid"] == kid:
            public_key_bytes = base64.urlsafe_b64decode(jwk["x"] + "==")
            return Ed25519PublicKey.from_public_bytes(public_key_bytes)
    raise ValueError(f"No matching JWK for kid={kid}")


def validate_neon_token(token: str) -> Optional[dict]:
    """Returns the decoded claims dict if the token is valid, else None.
    Claims include at least: sub (user id), email, name, iat, exp."""
    try:
        jwks = _get_jwks()
        try:
            signing_key = _get_signing_key(token, jwks)
        except ValueError:
            # Key rotated since our last fetch — refresh once and retry.
            jwks = _get_jwks(force=True)
            signing_key = _get_signing_key(token, jwks)

        return jwt.decode(
            token, key=signing_key, algorithms=["EdDSA"], issuer=ORIGIN, audience=ORIGIN
        )
    except PyJWTError as exc:
        print(f"[warning] Neon Auth token validation failed: {exc}")
        return None
    except Exception as exc:
        print(f"[warning] Unexpected error validating Neon Auth token: {exc}")
        return None