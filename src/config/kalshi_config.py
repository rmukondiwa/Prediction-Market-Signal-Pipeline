import os
from dataclasses import dataclass, field
from pathlib import Path


def _resolve_private_key() -> str:
    """
    Load the RSA private key PEM text from either:
      1. KALSHI_PRIVATE_KEY  — the raw key content pasted directly into .env
      2. KALSHI_PRIVATE_KEY_PATH — path to a .pem/.txt file containing the key
    Raises ValueError if neither is set.
    """
    raw = os.getenv("KALSHI_PRIVATE_KEY", "").strip()
    if raw:
        # Env vars can't store literal newlines easily; support \n as escape
        return raw.replace("\\n", "\n")

    path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if path:
        key_path = Path(path)
        if not key_path.exists():
            raise FileNotFoundError(f"Private key file not found: {path}")
        return key_path.read_text().strip()

    raise ValueError(
        "Kalshi private key not configured. "
        "Set KALSHI_PRIVATE_KEY (raw PEM text) or KALSHI_PRIVATE_KEY_PATH (file path)."
    )


@dataclass
class KalshiConfig:
    # Auth
    api_key_id: str = field(default_factory=lambda: os.environ["KALSHI_API_KEY_ID"])
    private_key_pem: str = field(default_factory=_resolve_private_key)

    # Endpoints
    ws_url: str = field(
        default_factory=lambda: os.getenv(
            "KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2"
        )
    )
    rest_base_url: str = field(
        default_factory=lambda: os.getenv(
            "KALSHI_REST_URL", "https://api.elections.kalshi.com/trade-api/v2"
        )
    )

    # Subscriptions
    market_tickers: list[str] = field(
        default_factory=lambda: [
            t.strip()
            for t in os.getenv("KALSHI_MARKET_TICKERS", "").split(",")
            if t.strip()
        ]
    )
    channels: list[str] = field(
        default_factory=lambda: [
            c.strip()
            for c in os.getenv("KALSHI_CHANNELS", "ticker,orderbook_delta,trade").split(",")
            if c.strip()
        ]
    )

    # Connection
    reconnect_attempts: int = field(
        default_factory=lambda: int(os.getenv("KALSHI_RECONNECT_ATTEMPTS", "10"))
    )
    reconnect_base_delay: float = field(
        default_factory=lambda: float(os.getenv("KALSHI_RECONNECT_BASE_DELAY", "1.0"))
    )
    reconnect_max_delay: float = field(
        default_factory=lambda: float(os.getenv("KALSHI_RECONNECT_MAX_DELAY", "60.0"))
    )
    ping_interval: int = field(
        default_factory=lambda: int(os.getenv("KALSHI_PING_INTERVAL", "20"))
    )
