from pydantic import BaseModel

from src.context.models import ContextMarket
from src.insight.models import MarketSnapshot


class DerivedProbability(BaseModel):
    description: str  # e.g. "P(oil spike | iran strike)"
    value: float      # 0.0 to 1.0
    reasoning: str


class Mispricing(BaseModel):
    ticker: str
    title: str
    direction: str              # "overpriced" or "underpriced"
    current_implied_prob: float
    estimated_fair_prob: float
    reasoning: str


class Edge(BaseModel):
    ticker: str
    title: str
    side: str           # "yes" or "no"
    confidence: str     # "low", "medium", "high"
    thesis: str
    kelly_fraction: float  # quarter-Kelly position size (0.0 to 0.25)


class InferenceReport(BaseModel):
    focus_market: MarketSnapshot
    context_markets: list[ContextMarket]
    consistency_analysis: str
    derived_probabilities: list[DerivedProbability]
    detected_mispricings: list[Mispricing]
    suggested_edges: list[Edge]
