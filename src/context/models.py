from src.catalog.models import CatalogMarket


class CandidateMarket(CatalogMarket):
    """A market returned by vector similarity search, before LLM reranking."""
    similarity_score: float


class ContextMarket(CatalogMarket):
    """A market that passed LLM reranking — includes causal relationship annotation."""
    similarity_score: float
    relevance_score: float  # 0-10, assigned by the LLM
    relationship: str       # LLM explanation of the causal/conditional link
