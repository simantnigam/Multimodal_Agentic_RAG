import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base
from core.constants import EMBEDDING_DIM_BGE_LARGE


class Asset(Base):
    __tablename__ = "assets"

    asset_id:  Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id:    Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    type:      Mapped[str | None]     = mapped_column(String)             # image | table
    content:   Mapped[str | None]     = mapped_column(String)             # alt-text, caption, or table markdown
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM_BGE_LARGE))
    metadata:  Mapped[dict | None]    = mapped_column(JSONB)

    document: Mapped["Document"] = relationship("Document", back_populates="assets")
