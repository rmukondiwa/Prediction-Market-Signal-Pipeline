import json
from enum import Enum
from typing import Any

from src.utils.logging import get_logger

logger = get_logger(__name__)


class MessageType(str, Enum):
    SUBSCRIBED = "subscribed"
    TICKER = "ticker"
    ORDERBOOK_SNAPSHOT = "orderbook_snapshot"
    ORDERBOOK_DELTA = "orderbook_delta"
    TRADE = "trade"
    ERROR = "error"
    UNKNOWN = "unknown"


class ParsedMessage:
    __slots__ = ("type", "msg", "seq", "sid")

    def __init__(
        self,
        type: MessageType,
        msg: dict[str, Any],
        seq: int = 0,
        sid: int = 0,
    ) -> None:
        self.type = type
        self.msg = msg
        self.seq = seq
        self.sid = sid

    def __repr__(self) -> str:
        return f"ParsedMessage(type={self.type}, seq={self.seq})"


class MessageParser:
    """
    Interprets raw JSON strings from the Kalshi WebSocket feed.

    Responsibilities:
    - Deserialize the raw JSON payload.
    - Identify the message type.
    - Return a ParsedMessage containing the typed enum and the inner `msg` dict.

    This class deliberately knows nothing about Kalshi-specific field semantics
    beyond the top-level `type` discriminator — that is the Normalizer's job.
    """

    def parse(self, raw: str) -> ParsedMessage | None:
        """
        Parse a raw WebSocket message string.

        Returns a ParsedMessage, or None if the message should be silently skipped
        (e.g. heartbeat pings).  Logs a warning and returns None on parse errors.
        """
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to decode JSON message",
                extra={"error": str(exc), "raw_preview": raw[:200]},
            )
            return None

        raw_type: str = data.get("type", "")
        try:
            msg_type = MessageType(raw_type)
        except ValueError:
            logger.debug(
                "Received unknown message type",
                extra={"raw_type": raw_type},
            )
            msg_type = MessageType.UNKNOWN

        msg: dict[str, Any] = data.get("msg", {})
        seq: int = data.get("seq", 0)
        sid: int = data.get("sid", 0)

        logger.debug(
            "Parsed message",
            extra={"type": msg_type.value, "seq": seq, "sid": sid},
        )
        return ParsedMessage(type=msg_type, msg=msg, seq=seq, sid=sid)
