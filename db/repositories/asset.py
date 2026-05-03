import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models.asset import Asset


async def insert_asset(session: AsyncSession, asset: Asset) -> Asset:
    session.add(asset)
    await session.flush()
    return asset


async def insert_assets(session: AsyncSession, assets: list[Asset]) -> list[Asset]:
    session.add_all(assets)
    await session.flush()
    return assets


async def get_asset(session: AsyncSession, asset_id: uuid.UUID) -> Asset | None:
    result = await session.execute(select(Asset).where(Asset.asset_id == asset_id))
    return result.scalar_one_or_none()


async def get_assets_by_document(session: AsyncSession, doc_id: uuid.UUID) -> list[Asset]:
    result = await session.execute(select(Asset).where(Asset.doc_id == doc_id))
    return list(result.scalars().all())


async def get_assets_by_ids(session: AsyncSession, asset_ids: list[uuid.UUID]) -> list[Asset]:
    result = await session.execute(select(Asset).where(Asset.asset_id.in_(asset_ids)))
    return list(result.scalars().all())
