"""
Qdrant RAG Service
Embeds historical GPU incidents into a vector store and retrieves
similar past incidents to augment the LangGraph diagnosis agent.

Architecture:
  - Embedding model: sentence-transformers/all-MiniLM-L6-v2 (local, no API cost)
  - Vector store: Qdrant (runs in Docker or in-memory for tests)
  - Retrieval: top-k cosine similarity on incident text

Day 2: called in fetch_context node to inject similar past incidents
into the LLM context before signal analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_client = None
_encoder = None


def _get_encoder():
    """Lazy-load the sentence transformer to avoid slow startup."""
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _encoder = SentenceTransformer(settings.embedding_model)
    return _encoder


def _get_client():
    """Lazy-load the Qdrant client."""
    global _client
    if _client is None:
        from qdrant_client import QdrantClient
        if settings.qdrant_enabled:
            logger.info(f"Connecting to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
            _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        else:
            # In-memory Qdrant — no Docker required, perfect for dev/test
            logger.info("Using in-memory Qdrant (qdrant_enabled=False)")
            _client = QdrantClient(":memory:")
    return _client


def _incident_to_text(incident: dict) -> str:
    """
    Flatten an incident dict into a single searchable text string.
    This is what gets embedded and searched.
    """
    return (
        f"severity={incident['severity']} "
        f"fix={incident['fix_category']} "
        f"node={incident.get('node', '')} "
        f"root_cause={incident['root_cause']} "
        f"factors={' '.join(incident.get('contributing_factors', []))} "
        f"log={incident.get('log_snippet', '')}"
    )


async def ensure_collection_seeded() -> bool:
    """
    Create the Qdrant collection and seed it with historical incidents
    from fixtures/incidents/historical_incidents.json.
    Idempotent — safe to call multiple times.
    Returns True if collection was seeded, False if already existed.
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = _get_client()
    collection = settings.qdrant_collection

    # Check if collection already has data
    try:
        info = client.get_collection(collection)
        if info.points_count and info.points_count > 0:
            logger.info(f"Qdrant collection '{collection}' already seeded ({info.points_count} points)")
            return False
    except Exception:
        pass  # Collection doesn't exist yet

    encoder = _get_encoder()
    dim = encoder.get_sentence_embedding_dimension()

    # Create collection
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    logger.info(f"Created Qdrant collection '{collection}' (dim={dim})")

    # Load historical incidents
    fixture_path = Path(settings.fixtures_dir) / "incidents" / "historical_incidents.json"
    incidents: list[dict] = json.loads(fixture_path.read_text())

    # Embed and upsert
    texts = [_incident_to_text(inc) for inc in incidents]
    vectors = encoder.encode(texts, show_progress_bar=False).tolist()

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "incident_id": inc["incident_id"],
                "date": inc["date"],
                "severity": inc["severity"],
                "fix_category": inc["fix_category"],
                "root_cause": inc["root_cause"],
                "contributing_factors": inc["contributing_factors"],
                "resolution_minutes": inc["resolution_minutes"],
                "log_snippet": inc.get("log_snippet", ""),
                "remediation_steps": inc.get("remediation_steps", []),
                "prevention": inc.get("prevention", ""),
                "node": inc.get("node", ""),
                "pod": inc.get("pod", ""),
            },
        )
        for i, inc in enumerate(incidents)
    ]

    client.upsert(collection_name=collection, points=points)
    logger.info(f"Seeded Qdrant with {len(points)} historical incidents")
    return True


async def retrieve_similar_incidents(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Embed `query` and retrieve the top-k most similar historical incidents.
    Returns a list of incident payload dicts, ranked by similarity.

    Called from the LangGraph fetch_context node to inject RAG context.
    """
    k = top_k or settings.rag_top_k
    client = _get_client()
    encoder = _get_encoder()

    await ensure_collection_seeded()

    query_vector = encoder.encode(query, show_progress_bar=False).tolist()

    try:
        response = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=k,
            with_payload=True,
        )
        hits = response.points
    except AttributeError:
        hits = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=query_vector,
            limit=k,
            with_payload=True,
        )

    similar = []
    for hit in hits:
        payload = dict(hit.payload or {})
        payload["similarity_score"] = round(hit.score, 3)
        similar.append(payload)
        logger.info(
            f"RAG hit: {payload.get('incident_id')} "
            f"(score={hit.score:.3f}, fix={payload.get('fix_category')})"
        )

    return similar


async def upsert_incident(incident_id: str, text: str, payload: dict) -> None:
    """
    Add a newly diagnosed incident to the vector store so future diagnoses
    can learn from it. Called at the end of run_diagnosis().
    """
    from qdrant_client.models import PointStruct

    client = _get_client()
    encoder = _get_encoder()

    await ensure_collection_seeded()

    # Use a hash of the incident_id as the point ID
    point_id = abs(hash(incident_id)) % (2**31)
    vector = encoder.encode(text, show_progress_bar=False).tolist()

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
    logger.info(f"Upserted new incident {incident_id} into Qdrant")
