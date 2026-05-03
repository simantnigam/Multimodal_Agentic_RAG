import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import String, Integer, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, TSVECTOR
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base
from core.constants import EMBEDDING_DIM_BGE_LARGE


class Chunk(Base):
    __tablename__ = "chunks"

    chunk_id:                Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id:                  Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    section_id:              Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("sections.section_id"), nullable=True)
    content:                 Mapped[str]            = mapped_column(String, nullable=False)
    embedding:               Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM_BGE_LARGE))
    embedding_model_version: Mapped[str | None]        = mapped_column(String)   # set from settings.embedding_model_version at insert time
    modality:                Mapped[str]            = mapped_column(String, default="text")   # text | image | table
    token_count:             Mapped[int | None]     = mapped_column(Integer)
    metadata:                Mapped[dict | None]    = mapped_column(JSONB)
    tsv:                     Mapped[str | None]     = mapped_column(TSVECTOR)                 # populated by DB trigger
    created_at:              Mapped[datetime]       = mapped_column(DateTime, server_default=func.now())

    document: Mapped["Document"] = relationship("Document", back_populates="chunks")
    section:  Mapped["Section | None"] = relationship("Section", back_populates="chunks")
