"""The bearer token that gates every state-mutating route.

Judgment calls this module fixes:

- One token, read from ``RISK_SCORING_API_TOKEN``. The service and the
  client read the same variable, so the replay harness, the batch
  scorer, and the check scripts pick it up from the shell they run in
  with no flag to forget.
- Unset and empty are the same thing: no token. The service refuses to
  start on either, the way it refuses to start without a model pin,
  because a service that silently ran open would be the defect this
  module exists to prevent.
- The comparison is ``hmac.compare_digest``, so a wrong token takes the
  same time to refuse whatever prefix it shares with the right one.
- The read-only routes stay open. ``/health`` is what the Compose
  healthcheck and the client's wait-for-ready poll, and ``/version``
  reveals nothing that the prediction log does not.
"""

from __future__ import annotations

import hmac
import os

ENV_API_TOKEN = "RISK_SCORING_API_TOKEN"

_SCHEME = "Bearer"


def api_token() -> str | None:
    """The token from the environment, or None when it is unset or empty."""
    token = os.environ.get(ENV_API_TOKEN, "").strip()
    return token or None


def require_api_token() -> str:
    """The token from the environment; raises naming the variable when absent."""
    token = api_token()
    if token is None:
        raise RuntimeError(
            f"{ENV_API_TOKEN} is not set; a bearer token is required to post events, "
            f"and the service refuses to start without one"
        )
    return token


def bearer_headers(token: str) -> dict[str, str]:
    """The header a caller sends to prove it holds the token."""
    return {"Authorization": f"{_SCHEME} {token}"}


def authorized(authorization: str | None, token: str) -> bool:
    """Whether an ``Authorization`` header value carries exactly ``token``."""
    if authorization is None:
        return False
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != _SCHEME.lower():
        return False
    return hmac.compare_digest(presented.strip().encode(), token.encode())
