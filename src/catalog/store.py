"""
Catalog persistence helpers.

Saves and loads the list of CatalogMarket records to/from disk.
Uses orjson when available for faster serialization, falls back to stdlib json.
"""

from pathlib import Path

from src.catalog.models import CatalogMarket
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PATH = Path("data/catalog.json")


def save_catalog(markets: list[CatalogMarket], path: Path = _DEFAULT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [m.model_dump(mode="json") for m in markets]

    try:
        import orjson
        path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    except ImportError:
        import json
        path.write_text(json.dumps(payload, indent=2))

    logger.info("Catalog saved", extra={"path": str(path), "count": len(markets)})


def load_catalog(path: Path = _DEFAULT_PATH) -> list[CatalogMarket]:
    try:
        import orjson
        raw = orjson.loads(path.read_bytes())
    except ImportError:
        import json
        raw = json.loads(path.read_text())

    catalog = [CatalogMarket(**item) for item in raw]
    logger.info("Catalog loaded", extra={"path": str(path), "count": len(catalog)})
    return catalog
