import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.chunk import Chunk


async def insert_chunk(session: AsyncSession, chunk: Chunk) -> Chunk:
    session.add(chunk)
    await session.flush()
    return chunk


async def insert_chunks(session: AsyncSession, chunks: list[Chunk]) -> list[Chunk]:
    session.add_all(chunks)
    await session.flush()
    return chunks


async def get_chunk(session: AsyncSession, chunk_id: uuid.UUID) -> Chunk | None:
    result = await session.execute(select(Chunk).where(Chunk.chunk_id == chunk_id))
    return result.scalar_one_or_none()


async def get_chunks_by_document(session: AsyncSession, doc_id: uuid.UUID) -> list[Chunk]:
    result = await session.execute(
        select(Chunk)
        .where(Chunk.doc_id == doc_id)
        .order_by(Chunk.created_at)
    )
    return list(result.scalars().all())


async def get_chunks_by_section(session: AsyncSession, section_id: uuid.UUID) -> list[Chunk]:
    result = await session.execute(
        select(Chunk)
        .where(Chunk.section_id == section_id)
        .order_by(Chunk.created_at)
    )
    return list(result.scalars().all())


async def get_chunks_by_ids(session: AsyncSession, chunk_ids: list[uuid.UUID]) -> list[Chunk]:
    result = await session.execute(select(Chunk).where(Chunk.chunk_id.in_(chunk_ids)))
    return list(result.scalars().all())
