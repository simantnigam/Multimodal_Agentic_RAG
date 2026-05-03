import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.document import Document


async def insert_document(session: AsyncSession, doc: Document) -> Document:
    session.add(doc)
    await session.flush()   # assigns doc_id without committing
    return doc


async def get_document(session: AsyncSession, doc_id: uuid.UUID) -> Document | None:
    result = await session.execute(select(Document).where(Document.doc_id == doc_id))
    return result.scalar_one_or_none()


async def get_all_documents(session: AsyncSession) -> list[Document]:
    result = await session.execute(select(Document).order_by(Document.created_at.desc()))
    return list(result.scalars().all())


async def delete_document(session: AsyncSession, doc_id: uuid.UUID) -> bool:
    doc = await get_document(session, doc_id)
    if doc is None:
        return False
    await session.delete(doc)
    return True
