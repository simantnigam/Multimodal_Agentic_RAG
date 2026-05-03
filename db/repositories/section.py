import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.section import Section


async def insert_section(session: AsyncSession, section: Section) -> Section:
    session.add(section)
    await session.flush()
    return section


async def insert_sections(session: AsyncSession, sections: list[Section]) -> list[Section]:
    session.add_all(sections)
    await session.flush()
    return sections


async def get_section(session: AsyncSession, section_id: uuid.UUID) -> Section | None:
    result = await session.execute(select(Section).where(Section.section_id == section_id))
    return result.scalar_one_or_none()


async def get_sections_by_document(session: AsyncSession, doc_id: uuid.UUID) -> list[Section]:
    result = await session.execute(
        select(Section)
        .where(Section.doc_id == doc_id)
        .order_by(Section.level)
    )
    return list(result.scalars().all())
