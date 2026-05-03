import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.relationship import Relationship


async def insert_relationship(session: AsyncSession, rel: Relationship) -> Relationship:
    session.add(rel)
    await session.flush()
    return rel


async def insert_relationships(session: AsyncSession, rels: list[Relationship]) -> list[Relationship]:
    session.add_all(rels)
    await session.flush()
    return rels


async def get_targets(session: AsyncSession, source_id: uuid.UUID) -> list[Relationship]:
    """Fetch all relationships where source_id matches — used by context assembly."""
    result = await session.execute(
        select(Relationship).where(Relationship.source_id == source_id)
    )
    return list(result.scalars().all())


async def get_sources(session: AsyncSession, target_id: uuid.UUID) -> list[Relationship]:
    """Fetch all relationships where target_id matches — reverse lookup."""
    result = await session.execute(
        select(Relationship).where(Relationship.target_id == target_id)
    )
    return list(result.scalars().all())


async def get_relationships_by_type(
    session: AsyncSession,
    source_id: uuid.UUID,
    relation_type: str,     # use RELATION_CHUNK_TO_ASSET or RELATION_CHUNK_TO_CHUNK from core.constants
) -> list[Relationship]:
    result = await session.execute(
        select(Relationship)
        .where(
            Relationship.source_id == source_id,
            Relationship.relation_type == relation_type,
        )
    )
    return list(result.scalars().all())
