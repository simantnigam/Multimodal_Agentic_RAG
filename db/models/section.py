import uuid

from sqlalchemy import String, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db.models.base import Base


class Section(Base):
    __tablename__ = "sections"

    section_id:         Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    doc_id:             Mapped[uuid.UUID]      = mapped_column(UUID(as_uuid=True), ForeignKey("documents.doc_id", ondelete="CASCADE"), nullable=False)
    parent_section_id:  Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    title:              Mapped[str | None]     = mapped_column(String)
    level:              Mapped[int | None]     = mapped_column(Integer)   # 1=H1, 2=H2, ...
    metadata:           Mapped[dict | None]    = mapped_column(JSONB)

    document: Mapped["Document"] = relationship("Document", back_populates="sections")
    chunks:   Mapped[list["Chunk"]] = relationship("Chunk", back_populates="section")
