import uuid

from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base


class Relationship(Base):
    __tablename__ = "relationships"

    id:            Mapped[int]       = mapped_column(Integer, primary_key=True)
    source_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)   # chunk_id or asset_id
    target_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)   # chunk_id or asset_id
    relation_type: Mapped[str]       = mapped_column(String, nullable=False)               # chunk_to_asset | chunk_to_chunk
