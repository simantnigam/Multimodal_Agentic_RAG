from dataclasses import dataclass, field
from typing import Any

import numpy as np
import tiktoken

from core.config import get_settings
from core.constants import MODALITY_TEXT
from core.interfaces.embedder import Embedder
from core.logging import get_logger

logger = get_logger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------

@dataclass
class ChunkData:
    content:     str
    token_count: int
    modality:    str = MODALITY_TEXT
    metadata:    dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_blocks(
    blocks:        list[str],
    section_title: str,
    embedder:      Embedder,
    modality:      str = MODALITY_TEXT,
) -> list[ChunkData]:
    """
    Convert a flat list of text blocks into semantically coherent chunks.

    Steps:
      1. Embed all blocks with the same model used for retrieval.
      2. Split at consecutive similarity < threshold (semantic boundary detection).
      3. Enforce token bounds: split oversized chunks, merge undersized ones.
      4. Inject section_title into chunk content for retrieval context.
      5. Attach full metadata schema (section_title, prev_chunk_summary,
         linked_elements, semantic_tags, chunk_summary placeholders).

    Args:
        blocks:        Pre-parsed text blocks from ingestion/parser.py.
        section_title: Section heading these blocks belong to.
        embedder:      Embedder instance — must be the same model used at query time.
        modality:      text | image | table (default: text).

    Returns:
        List of ChunkData ready to be stored in the chunks table.
    """
    if not blocks:
        return []

    settings = get_settings()
    threshold  = settings.chunking_similarity_threshold
    min_tokens = settings.chunk_min_tokens
    max_tokens = settings.chunk_max_tokens

    # Reserve token budget for the context header ([Section Title]\n\n)
    # so that enforced chunks never exceed max_tokens after injection.
    header_tokens  = _count_tokens(_inject_context("", section_title))
    effective_max  = max(min_tokens + 1, max_tokens - header_tokens)

    # Single block — skip similarity computation
    if len(blocks) == 1:
        return _to_chunks([blocks[0]], section_title, modality, effective_max, min_tokens)

    # Embed all blocks to detect semantic boundaries
    embeddings = np.array(embedder.embed_texts(blocks), dtype=np.float32)

    # Compute consecutive cosine similarities (L2-normalised vectors → dot = cosine)
    similarities = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]

    # Split blocks into raw chunks at semantic boundaries
    raw_chunks = _split_at_boundaries(blocks, similarities, threshold)

    logger.debug(
        "chunking_boundaries",
        section=section_title,
        blocks=len(blocks),
        raw_chunks=len(raw_chunks),
        threshold=threshold,
    )

    return _to_chunks(raw_chunks, section_title, modality, effective_max, min_tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_at_boundaries(
    blocks:       list[str],
    similarities: list[float],
    threshold:    float,
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = [blocks[0]]

    for i, sim in enumerate(similarities):
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [blocks[i + 1]]
        else:
            current.append(blocks[i + 1])

    chunks.append(" ".join(current))
    return chunks


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _first_sentence(text: str) -> str:
    """Return the first sentence of a text — used as a lightweight prev_chunk_summary."""
    end = text.find(". ")
    return text[: end + 1].strip() if end != -1 else text[:120].strip()


def _inject_context(content: str, section_title: str) -> str:
    """
    Prepend section title to chunk content so retrievers and the LLM
    always know which section a chunk belongs to without a DB lookup.
    """
    return f"[{section_title}]\n\n{content}"


def _split_oversized(text: str, max_tokens: int) -> list[str]:
    """Split a chunk that exceeds max_tokens at sentence boundaries."""
    sentences = text.split(". ")
    parts: list[str] = []
    current_tokens = 0
    current: list[str] = []

    for sent in sentences:
        sent_tokens = _count_tokens(sent)
        if current_tokens + sent_tokens > max_tokens and current:
            parts.append(". ".join(current).strip())
            current = [sent]
            current_tokens = sent_tokens
        else:
            current.append(sent)
            current_tokens += sent_tokens

    if current:
        parts.append(". ".join(current).strip())

    return [p for p in parts if p]


def _enforce_bounds(
    raw_chunks: list[str],
    max_tokens: int,
    min_tokens: int,
) -> list[str]:
    """Split oversized chunks, then merge undersized ones with their neighbour."""
    # Step 1: split oversized
    sized: list[str] = []
    for chunk in raw_chunks:
        if _count_tokens(chunk) > max_tokens:
            sized.extend(_split_oversized(chunk, max_tokens))
        else:
            sized.append(chunk)

    # Step 2: merge undersized into the next chunk
    merged: list[str] = []
    i = 0
    while i < len(sized):
        if _count_tokens(sized[i]) < min_tokens and i + 1 < len(sized):
            merged.append(sized[i] + " " + sized[i + 1])
            i += 2
        else:
            merged.append(sized[i])
            i += 1

    return merged


def _to_chunks(
    raw_chunks:    list[str],
    section_title: str,
    modality:      str,
    max_tokens:    int,
    min_tokens:    int,
) -> list[ChunkData]:
    enforced = _enforce_bounds(raw_chunks, max_tokens, min_tokens)
    result: list[ChunkData] = []

    for idx, raw_text in enumerate(enforced):
        # Inject section title into content for context-aware retrieval
        content = _inject_context(raw_text, section_title)
        tokens  = _count_tokens(content)

        # prev_chunk_summary: first sentence of the preceding chunk
        # (lightweight placeholder — replaced by LLM-generated summary in Phase 4)
        if idx > 0:
            prev_summary = _first_sentence(enforced[idx - 1])
        else:
            prev_summary = ""

        result.append(ChunkData(
            content=content,
            token_count=tokens,
            modality=modality,
            metadata={
                "section_title":     section_title,
                "chunk_index":       idx,
                "prev_chunk_summary": prev_summary,
                "linked_elements":   [],       # populated by ingestion/pipeline.py during storage
                "semantic_tags":     [],       # populated by NLP/LLM in Phase 4+
                "chunk_summary":     "",       # populated by LLM in Phase 4+
            },
        ))

    return result
