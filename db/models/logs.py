import uuid
from datetime import datetime

from sqlalchemy import Integer, Float, String, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from db.models.base import Base


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id:                  Mapped[int]            = mapped_column(Integer, primary_key=True)
    trace_id:            Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    query:               Mapped[str | None]     = mapped_column(Text)
    retrieved_chunk_ids: Mapped[list | None]    = mapped_column(ARRAY(UUID(as_uuid=True)))
    scores:              Mapped[list | None]    = mapped_column(ARRAY(Float))
    latency_ms:          Mapped[int | None]     = mapped_column(Integer)
    created_at:          Mapped[datetime]       = mapped_column(DateTime, server_default=func.now())


class EvaluationLog(Base):
    __tablename__ = "evaluation_logs"

    id:                  Mapped[int]            = mapped_column(Integer, primary_key=True)
    trace_id:            Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    query:               Mapped[str | None]     = mapped_column(Text)
    answer:              Mapped[str | None]     = mapped_column(Text)
    groundedness_score:  Mapped[float | None]   = mapped_column(Float)
    completeness_score:  Mapped[float | None]   = mapped_column(Float)
    hallucination_score: Mapped[float | None]   = mapped_column(Float)
    final_score:         Mapped[float | None]   = mapped_column(Float)
    retry_count:         Mapped[int]            = mapped_column(Integer, default=0)
    confidence:          Mapped[float | None]   = mapped_column(Float)
    latency_breakdown:   Mapped[dict | None]    = mapped_column(JSONB)
    feedback:            Mapped[str | None]     = mapped_column(Text)
    created_at:          Mapped[datetime]       = mapped_column(DateTime, server_default=func.now())
