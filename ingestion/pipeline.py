from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from core.constants import (
    MODALITY_IMAGE,
    MODALITY_TABLE,
    RELATION_CHUNK_TO_ASSET,
)
from core.config import get_settings
from core.logging import get_logger
from chunking.engine import chunk_blocks
from db.models.document import Document
from db.models.section import Section
from db.models.chunk import Chunk
from db.models.asset import Asset
from db.models.relationship import Relationship
from db.repositories.document import insert_document
from db.repositories.section import insert_section
from db.repositories.chunk import insert_chunks
from db.repositories.asset import insert_assets
from db.repositories.relationship import insert_relationships
from embeddings import get_embedder
from embeddings.cache import embed_with_cache
from ingestion.parser import parse_document

logger = get_logger(__name__)


async def ingest_document(
    path:         str | Path,
    metadata:     dict[str, Any],
    session:      AsyncSession,
    redis_client: redis.Redis | None = None,
) -> str:
    """
    Full ingestion pipeline: parse → chunk → embed → store.

    Args:
        path:         Path to the document file.
        metadata:     Extra metadata to store on the document record (e.g. domain, tags).
        session:      Async DB session — caller owns the transaction boundary.
        redis_client: Optional Redis client for embedding cache. Pass None to skip cache.

    Returns:
        doc_id of the stored document (UUID string).

    Raises:
        Any exception from parsing, embedding, or DB — caller should rollback on error.
        The pipeline itself does not commit; the session generator in db/session.py handles that.
    """
    settings = get_settings()
    embedder = get_embedder()
    path     = Path(path)

    # ------------------------------------------------------------------
    # Step 1: Parse
    # ------------------------------------------------------------------
    logger.info("ingestion_start", path=str(path))
    parsed = parse_document(path)

    # ------------------------------------------------------------------
    # Step 2: Insert document
    # ------------------------------------------------------------------
    doc_id = str(uuid.uuid4())
    doc    = Document(
        doc_id=uuid.UUID(doc_id),
        title=parsed.title,
        source=parsed.source,
        doc_type=parsed.doc_type,
        metadata={**parsed.metadata, **metadata},
    )
    await insert_document(session, doc)
    logger.info("document_inserted", doc_id=doc_id, title=parsed.title)

    # ------------------------------------------------------------------
    # Step 3: Process each section — chunk + embed + store
    # ------------------------------------------------------------------
    # Track the most recent section's chunk IDs to link nearby assets
    last_chunk_ids: list[uuid.UUID] = []

    for parsed_section in parsed.sections:
        if not parsed_section.blocks:
            continue

        # Insert section record
        section_id  = uuid.uuid4()
        section_orm = Section(
            section_id=section_id,
            doc_id=uuid.UUID(doc_id),
            title=parsed_section.title,
            level=parsed_section.level,
        )
        await insert_section(session, section_orm)

        # Chunk the text blocks
        chunk_data_list = chunk_blocks(
            blocks=parsed_section.blocks,
            section_title=parsed_section.title,
            embedder=embedder,
        )

        if not chunk_data_list:
            continue

        # Embed chunks (with cache if Redis available)
        contents = [cd.content for cd in chunk_data_list]
        if redis_client is not None:
            embeddings = embed_with_cache(contents, embedder, redis_client)
        else:
            embeddings = embedder.embed_texts(contents)

        # Build ORM Chunk objects
        chunk_orms: list[Chunk] = []
        chunk_ids:  list[uuid.UUID] = []
        for cd, embedding in zip(chunk_data_list, embeddings):
            chunk_id = uuid.uuid4()
            chunk_ids.append(chunk_id)
            chunk_orms.append(Chunk(
                chunk_id=chunk_id,
                doc_id=uuid.UUID(doc_id),
                section_id=section_id,
                content=cd.content,
                embedding=embedding,
                embedding_model_version=settings.embedding_model_version,
                modality=cd.modality,
                token_count=cd.token_count,
                metadata=cd.metadata,
            ))

        await insert_chunks(session, chunk_orms)
        last_chunk_ids = chunk_ids

        logger.debug(
            "section_ingested",
            section=parsed_section.title,
            chunks=len(chunk_orms),
        )

    # ------------------------------------------------------------------
    # Step 4: Process assets (images + tables) — embed + store + link
    # ------------------------------------------------------------------
    all_assets:       list[Asset]        = []
    all_relationships: list[Relationship] = []

    # Images: embed caption, store asset, link to last section's chunks
    if parsed.images:
        captions = [img.caption for img in parsed.images]
        img_embeddings = (
            embed_with_cache(captions, embedder, redis_client)
            if redis_client else embedder.embed_texts(captions)
        )

        for img, embedding in zip(parsed.images, img_embeddings):
            asset_id = uuid.uuid4()
            all_assets.append(Asset(
                asset_id=asset_id,
                doc_id=uuid.UUID(doc_id),
                type=MODALITY_IMAGE,
                content=img.caption,
                embedding=embedding,
                metadata={"page": img.page, "format": img.fmt},
            ))
            # Link to first chunk of the nearest section
            if last_chunk_ids:
                all_relationships.append(Relationship(
                    source_id=last_chunk_ids[0],
                    target_id=asset_id,
                    relation_type=RELATION_CHUNK_TO_ASSET,
                ))

    # Tables: embed markdown content, store asset, link to last section's chunks
    if parsed.tables:
        table_contents = [tab.content for tab in parsed.tables]
        tab_embeddings = (
            embed_with_cache(table_contents, embedder, redis_client)
            if redis_client else embedder.embed_texts(table_contents)
        )

        for tab, embedding in zip(parsed.tables, tab_embeddings):
            asset_id = uuid.uuid4()
            all_assets.append(Asset(
                asset_id=asset_id,
                doc_id=uuid.UUID(doc_id),
                type=MODALITY_TABLE,
                content=tab.content,
                embedding=embedding,
                metadata={"page": tab.page},
            ))
            if last_chunk_ids:
                all_relationships.append(Relationship(
                    source_id=last_chunk_ids[0],
                    target_id=asset_id,
                    relation_type=RELATION_CHUNK_TO_ASSET,
                ))

    if all_assets:
        await insert_assets(session, all_assets)

    if all_relationships:
        await insert_relationships(session, all_relationships)

        # Update linked_elements in chunk metadata to reflect stored asset UUIDs.
        # Group asset UUIDs by source chunk_id from the relationships list.
        chunk_asset_map: dict[str, list[str]] = {}
        for rel in all_relationships:
            key = str(rel.source_id)
            chunk_asset_map.setdefault(key, []).append(str(rel.target_id))

        for chunk_id_str, asset_ids in chunk_asset_map.items():
            await session.execute(
                text("""
                    UPDATE chunks
                    SET metadata = jsonb_set(
                        COALESCE(metadata, '{}'),
                        '{linked_elements}',
                        :asset_ids::jsonb
                    )
                    WHERE chunk_id = :chunk_id::uuid
                """),
                {"chunk_id": chunk_id_str, "asset_ids": json.dumps(asset_ids)},
            )

    logger.info(
        "ingestion_complete",
        doc_id=doc_id,
        sections=len(parsed.sections),
        assets=len(all_assets),
        relationships=len(all_relationships),
    )

    return doc_id
