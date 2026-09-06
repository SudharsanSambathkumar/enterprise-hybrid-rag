"""Tiny role-based access control used to filter retrieved content.

Every document declares `access: [...]` in its front matter. A user can see a
chunk if the chunk is open to `all`, if any of the user's roles is listed, or
if the user has the `admin` role.
"""

from __future__ import annotations

from collections.abc import Iterable

DEFAULT_ROLES = ("staff",)


def normalize_roles(roles: Iterable[str] | None) -> set[str]:
    return {r.strip().lower() for r in (roles or DEFAULT_ROLES) if r and r.strip()}


def can_access(access: Iterable[str], roles: Iterable[str] | None) -> bool:
    user = normalize_roles(roles)
    allowed = {a.strip().lower() for a in access}
    return "admin" in user or "all" in allowed or bool(allowed & user)
