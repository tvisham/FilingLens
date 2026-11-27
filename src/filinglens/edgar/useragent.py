"""SEC requires automated clients to declare a User-Agent with contact details."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"[^@\s<>]+@[^@\s<>]+\.[^@\s<>]+")


class UserAgentError(ValueError):
    """Raised when the configured User-Agent would be rejected by SEC."""


def validate_user_agent(value: str) -> str:
    """Return the trimmed User-Agent, or raise if it lacks an app name and an email."""
    ua = value.strip()
    if not _EMAIL.search(ua):
        raise UserAgentError(
            "SEC_USER_AGENT must include a contact email, e.g. 'FilingLens Jane Doe jane@example.edu'"
        )
    if " " not in ua:
        raise UserAgentError("SEC_USER_AGENT must include an app or person name before the email")
    return ua
