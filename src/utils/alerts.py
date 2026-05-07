"""
Live alerting — webhook-based notifications for high-severity events.

Routes alerts to one or more sinks:
  - Discord webhook (free, set DISCORD_WEBHOOK_URL)
  - Slack incoming webhook (free, set SLACK_WEBHOOK_URL)
  - Pushover (paid, set PUSHOVER_TOKEN + PUSHOVER_USER)
  - File sink (always-on, writes to logs/alerts.jsonl)

Use:
    alert("Kill switch tripped", severity="high",
          context={"realized_pnl": -302.79, "limit": 300})

Severity levels:
  - "info"     → file sink only
  - "warning"  → file sink + webhooks
  - "high"     → file sink + webhooks + Pushover (if configured)

Webhook posts are non-blocking (fire-and-forget via asyncio.create_task).
File sink is synchronous but cheap. Failure to deliver is logged but never
raises — alerts are best-effort, not blocking the trading loop.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import aiohttp

from src.utils.logging import get_logger

logger = get_logger(__name__)

Severity = Literal["info", "warning", "high"]
ALERT_LOG = Path("logs/alerts.jsonl")


def _file_sink(record: dict) -> None:
    ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ALERT_LOG.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        logger.warning("Alert file sink failed", extra={"error": str(e)})


async def _post_json(url: str, payload: dict, timeout: float = 8.0) -> bool:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload,
                                     timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                return r.status < 400
    except Exception as e:
        logger.warning("Alert webhook post failed",
                       extra={"url_host": url.split("/")[2] if "//" in url else "?",
                              "error": str(e)})
        return False


def _format_message(title: str, severity: Severity,
                     context: dict | None) -> str:
    icon = {"info": "ℹ️", "warning": "⚠️", "high": "🚨"}.get(severity, "•")
    parts = [f"{icon} **{title}**", f"Severity: {severity}"]
    if context:
        parts.append("```")
        for k, v in context.items():
            parts.append(f"  {k}: {v}")
        parts.append("```")
    parts.append(f"_{datetime.now(timezone.utc).isoformat()}_")
    return "\n".join(parts)


async def _discord(message: str) -> bool:
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        return False
    return await _post_json(url, {"content": message[:1900]})


async def _slack(message: str) -> bool:
    url = os.environ.get("SLACK_WEBHOOK_URL", "")
    if not url:
        return False
    return await _post_json(url, {"text": message})


async def _pushover(title: str, message: str, severity: Severity) -> bool:
    token = os.environ.get("PUSHOVER_TOKEN", "")
    user = os.environ.get("PUSHOVER_USER", "")
    if not (token and user):
        return False
    priority = 1 if severity == "high" else 0
    return await _post_json("https://api.pushover.net/1/messages.json", {
        "token": token, "user": user, "title": title, "message": message,
        "priority": priority,
    })


async def alert_async(title: str, severity: Severity = "warning",
                      context: dict | None = None) -> None:
    """Fire an alert to all configured sinks. Always lands in file sink.

    Webhook delivery is concurrent and best-effort — slow/down sinks don't
    block trading. Use this from async code.
    """
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "severity": severity,
        "context": context or {},
    }
    _file_sink(record)

    if severity == "info":
        return  # info goes to file only

    msg = _format_message(title, severity, context)
    tasks: list[Any] = [_discord(msg), _slack(msg)]
    if severity == "high":
        tasks.append(_pushover(title, msg, severity))
    # Fire-and-forget — don't await individual results, but do gather to
    # avoid leaking unawaited tasks
    await asyncio.gather(*tasks, return_exceptions=True)


def alert(title: str, severity: Severity = "warning",
          context: dict | None = None) -> None:
    """Sync wrapper. Schedules the async alert if a loop is running, else
    just hits the file sink (no webhooks)."""
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "title": title,
        "severity": severity,
        "context": context or {},
    }
    _file_sink(record)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(alert_async(title, severity, context))
    except RuntimeError:
        # No running loop — file sink is the best we can do synchronously
        pass
