import uuid
from datetime import datetime

from sqlalchemy import String, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base


class Document(Base):
    __tablename__ = "documents"

    doc_id:     Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title:      Mapped[str | None] = mapped_column(String)
    source:     Mapped[str | None] = mapped_column(String)
    doc_type:   Mapped[str | None] = mapped_column(String)       # pdf | html | docx | image
    created_at: Mapped[datetime]   = mapped_column(DateTime, server_default=func.now())
    metadata:   Mapped[dict | None] = mapped_column(JSONB)

    sections:      Mapped[list["Section"]]      = relationship("Section",      back_populates="document", cascade="all, delete-orphan")
    chunks:        Mapped[list["Chunk"]]        = relationship("Chunk",        back_populates="document", cascade="all, delete-orphan")
    assets:        Mapped[list["Asset"]]        = relationship("Asset",        back_populates="document", cascade="all, delete-orphan")
