"""Read-only, single-user Spendee MCP server for Render + GitHub OAuth.

Add this file to the root of a fork of https://github.com/gibsn/mcp-spendee.
Dependency: fastmcp>=2.13,<3 in addition to the fork's dependencies.
Not formally end-to-end tested against a live Spendee account.
"""
from __future__ import annotations

import os
from typing import Any

import anyio
from fastmcp import FastMCP
from fastmcp.server.auth.providers.github import GitHubProvider
from fastmcp.server.dependencies import get_access_token

from mcp_spendee.client import ResourceId, SpendeeGateway, TransactionType
from mcp_spendee.config import Settings


def env(name: str) -> str:
    """Require configuration and fail closed if it is missing."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required setting: {name}")
    return value


_auth = GitHubProvider(
    client_id=env("GITHUB_CLIENT_ID"),
    client_secret=env("GITHUB_CLIENT_SECRET"),
    base_url=env("PUBLIC_BASE_URL").rstrip("/"),
    jwt_signing_key=env("JWT_SIGNING_KEY"),
)

mcp = FastMCP(name="Spendee — private read-only", auth=_auth)
_gateway = SpendeeGateway(Settings.from_env())


def require_owner() -> None:
    """Only the specified GitHub username may call ANY Spendee tool.

    Authenticated GitHub users are NOT automatically allowed: identity is
    verified on every tool invocation, preventing other GitHub users from
    accessing financial records. Fail closed if the claim is missing.
    """
    token = get_access_token()
    claims = token.claims if token is not None else {}
    login = str(claims.get("login") or "")
    allowed_login = env("GITHUB_ALLOWED_LOGIN")
    if not login or login.casefold() != allowed_login.casefold():
        raise PermissionError("Access denied for this GitHub identity")


@mcp.tool()
async def spendee_status() -> dict[str, object]:
    """Check Spendee connection configuration; does not expose credentials."""
    require_owner()
    return Settings.from_env().status()


@mcp.tool()
async def list_wallets() -> list[dict[str, Any]]:
    """List Spendee wallets with wallet IDs, currencies and names."""
    require_owner()
    return await anyio.to_thread.run_sync(_gateway.list_wallets)


@mcp.tool()
async def list_labels() -> list[dict[str, str]]:
    """List existing Spendee labels."""
    require_owner()
    return await anyio.to_thread.run_sync(_gateway.list_labels)


@mcp.tool()
async def list_categories(
    wallet_id: ResourceId | None = None,
    category_type: TransactionType | None = None,
) -> list[dict[str, Any]]:
    """List existing Spendee categories."""
    require_owner()
    return await anyio.to_thread.run_sync(
        lambda: _gateway.list_categories(
            wallet_id=wallet_id,
            category_type=category_type,
        )
    )


@mcp.tool()
async def list_transactions(
    wallet_id: ResourceId | None = None,
    offset: int = 0,
    limit: int | None = 100,
    include_labels: bool = False,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[dict[str, Any]]:
    """Read transactions only; date bounds use UTC days/offset-aware timestamps.

    For a complete bounded period set both dates and limit=null.
    """
    require_owner()
    return await anyio.to_thread.run_sync(
        lambda: _gateway.list_transactions(
            wallet_id=wallet_id,
            offset=offset,
            limit=limit,
            include_labels=include_labels,
            date_from=date_from,
            date_to=date_to,
        )
    )


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
    )
