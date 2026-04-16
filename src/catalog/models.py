from pydantic import BaseModel


class CatalogMarket(BaseModel):
    """
    Merged record of a single Kalshi market and its parent event metadata.
    Canonical unit throughout the catalog, embedding, and retrieval stages.
    """
    ticker: str                 # Unique market identifier e.g. "INXD-23DEC29-B4000"
    event_ticker: str           # Parent event e.g. "INXD-23DEC29"
    title: str                  # Human-readable title from parent event
    subtitle: str               # Market-level subtitle / resolution rule
    category: str               # Kalshi category tag e.g. "Politics", "Finance"
    status: str                 # "open", "closed", "settled", etc.
    yes_bid: int                # Best yes bid in cents
    yes_ask: int                # Best yes ask in cents
    implied_probability: float  # (yes_bid + yes_ask) / 2 / 100
