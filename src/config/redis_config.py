import os
from dataclasses import dataclass, field


@dataclass
class RedisConfig:
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))
    password: str | None = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))

    # Stream names
    market_events_stream: str = field(
        default_factory=lambda: os.getenv("REDIS_MARKET_EVENTS_STREAM", "market_events")
    )
    trade_events_stream: str = field(
        default_factory=lambda: os.getenv("REDIS_TRADE_EVENTS_STREAM", "trade_events")
    )
    orderbook_events_stream: str = field(
        default_factory=lambda: os.getenv("REDIS_ORDERBOOK_EVENTS_STREAM", "orderbook_events")
    )

    # Stream max length (0 = unlimited)
    stream_max_len: int = field(
        default_factory=lambda: int(os.getenv("REDIS_STREAM_MAX_LEN", "100000"))
    )

    @property
    def url(self) -> str:
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"
