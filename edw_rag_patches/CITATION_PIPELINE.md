# Citation Pipeline and Block-Level Highlighting

This document describes how LightRAG turns parsed document content into inline
citations and PDF highlight coordinates.

The central distinction is:

- A **block** is the parser's visual/source unit. It can have page and bounding
  box positions and is what the viewer highlights.
- A **chunk** is an embedding and retrieval unit. A chunk can include several
  blocks; one block can also be split across several chunks.

Do not treat a chunk as a visual highlight target. A citation must ultimately
resolve to one or more parser blocks, preferably to a character range inside a
block.

## Citation syntax

The generated-answer format is:

```text
[^Index:DocumentID§SectionID¶SubchunkID]
```

Example:

```text
The pledge targets net-zero emissions by 2040[^1:doc-5501c9cdfc7b318b264a674230dfa432-chunk-000§root¶287cc6a64ac815058d91f76aa46e9e84].
```

The fields mean:

| Field | Meaning | Current source |
| --- | --- | --- |
| `Index` | Display number in the answer. It is not a storage key. | Query reference ordering. |
| `DocumentID` | Currently the retrieved `chunk_id`, for example `doc-…-chunk-000`. It lets legacy highlight storage find the retrieved evidence. | `text_chunks` / retrieval result. |
| `SectionID` | Structural section selector. | Currently `root` when no durable parser section ID exists. |
| `SubchunkID` | Parser block ID. This is the exact visual evidence target. | `blocks.jsonl` `blockid`. |

The name `DocumentID` is retained to match the citation grammar, but it is
currently a **retrieval chunk ID**, not `full_doc_id`. This is important when
resolving a citation programmatically.

### What `§root` means

`§root` is a stable placeholder for “no parser section identifier is currently
available.” It does **not** mean the whole PDF, the PDF root page, or all blocks
in the document. The precise selector is still the following `¶<blockid>`.

Current sidecar rows store a heading and its parent headings, but not a stable
section ID. Until one is added, a viewer should use `¶SubchunkID` to locate the
block and regard `§root` as informational only.

## End-to-end flow

```text
source PDF/DOCX/Markdown
        │
        ▼
parser IR → blocks.jsonl
        │       └─ content block: blockid, content, heading, positions[]
        ▼
chunker → embedding chunks
        │       └─ sidecar.refs: block IDs and, when available, text ranges
        ▼
text_chunks + chunks_vdb
        │
        ▼
retrieval → restore durable provenance from text_chunks
        │
        ▼
LLM context → citation_targets on each retrieved chunk
        │
        ▼
LLM answer → [^Index:ChunkID§SectionID¶BlockID]
        │
        ▼
EDW response filter → citation_highlights.sources
        │
        ▼
viewer → block/page/bbox highlight
```

## 1. Parsing: create visual blocks

Sidecar-capable parsers write `<document>.blocks.jsonl`. Each `type: "content"`
row contains, at minimum:

```json
{
  "type": "content",
  "blockid": "287cc6a64ac815058d91f76aa46e9e84",
  "content": "The source text for one visual block.",
  "heading": "Climate Pledge",
  "parent_headings": ["Sustainability"],
  "positions": [
    {
      "type": "bbox",
      "anchor": "1",
      "range": [72, 144, 523, 160]
    }
  ]
}
```

`positions` may have several boxes because one logical block can span columns,
lines, or pages. A block can also have no usable bounding box; this is normal
for parsers/formats that do not expose layout geometry.

The sidecar writer builds `blockid` from document identity, source order,
heading, and rendered content. It is stable only while those inputs are stable.
Re-parsing after materially changing the parser output can create new block IDs.

## 2. Chunking: map retrieval chunks back to blocks

The chunker receives merged block text. Chunk provenance is stored in its
`sidecar` field:

```json
{
  "type": "block",
  "id": "287cc6a64ac815058d91f76aa46e9e84",
  "refs": [
    {
      "type": "block",
      "id": "287cc6a64ac815058d91f76aa46e9e84",
      "start": 25,
      "end": 184
    }
  ]
}
```

`start` and `end` are half-open character offsets relative to that block's
`content`. They allow one block split into multiple chunks to remain
distinguishable. They are not PDF glyph coordinates.

### Chunker behaviour

- **Paragraph-semantic (`P`)** chunking starts from parser blocks and normally
  carries block IDs directly.
- **Fixed-token (`F`)**, **recursive (`R`)**, and **semantic-vector (`V`)**
  chunking work over merged text. `backfill_chunk_sidecars()` maps their source
  spans back to overlapping blocks after chunking.
- A chunk spanning several blocks has several `sidecar.refs`. The LLM must cite
  only the block supporting its claim, not all of them.
- Multimodal chunks may point at table, drawing, or equation sidecar items.
  Only `type: "block"` references currently become ordinary PDF block citation
  targets.

### Important limitation: exact quotes

The character offsets identify the chunk-derived portion of a block, but the
LLM's `<mark>…</mark>` text is still generated text. The UI should validate that
the marked text is an exact substring of the cited block before treating it as a
direct quote highlight. If it is not, degrade to block-level highlighting or
show no quote-specific highlight.

## 3. Storage

Two stores matter:

- `text_chunks` is authoritative for durable chunk fields such as `full_doc_id`,
  `heading`, and `sidecar` provenance.
- `chunks_vdb` is optimized for semantic search and may return only compact
  metadata such as content, file path, and chunk ID.

After vector retrieval, `operate._attach_content_headings()` fetches the
matching `text_chunks` rows to restore `document_id` and `sidecar`. This happens
before reranking and token truncation in both graph-backed and naive retrieval.

Raw text inserted without a sidecar has no parser blocks or bounding boxes. It
can still be cited at file/chunk level, but cannot produce a precise PDF overlay.

## 4. LLM context and answer generation

Each retained document chunk can include `citation_targets` in the LLM context:

```json
{
  "reference_id": "1",
  "content": "…",
  "citation_targets": [
    {
      "document_id": "doc-…-chunk-000",
      "section_id": "root",
      "subchunk_id": "287cc6a64ac815058d91f76aa46e9e84",
      "start": 25,
      "end": 184
    }
  ]
}
```

The prompts require inline citations immediately after the supporting claim.
They also prohibit a generic bibliography/reference section because it cannot
express exact visual evidence.

For a direct quotation, the required answer form is:

```text
The report states <mark>exact words from the block</mark> [^1:doc-…-chunk-000§root¶287cc…].
```

Only the text inside `mark` is intended to be an exact source substring.

## 5. API response provenance

`/query/data` returns retrieved chunks with their durable provenance where
available:

```json
{
  "chunk_id": "doc-…-chunk-000",
  "document_id": "doc-…-chunk-000",
  "sidecar": {"type": "block", "id": "…", "refs": ["…"]}
}
```

`/query` and `/query/stream` retain file-level `references` for compatibility.
They also attach compact `provenance` entries to each reference. A viewer should
use that provenance or `/query/data`; it should not infer a PDF location from a
file path alone.

## 6. EDW `citation_highlights` sidecar

The EDW patch builds a viewer-oriented sidecar:

```json
{
  "citation_highlights": {
    "version": 1,
    "sources": {
      "report.pdf": {
        "file_path": "report.pdf",
        "reference_id": "doc-…-chunk-000",
        "chunks": [
          {
            "chunk_id": "doc-…-chunk-000",
            "reference_id": "doc-…-chunk-000",
            "highlights": [
              {"page": 1, "bbox": {"l": 72, "t": 144, "r": 523, "b": 160}}
            ]
          }
        ]
      }
    }
  }
}
```

The EDW wrapper filters this sidecar to cited evidence after the LLM responds.
For `[^Index:ChunkID§SectionID¶BlockID]`, it must filter using **ChunkID**, not
`Index`. For example, in:

```text
[^1:doc-5501…-chunk-000§root¶287cc…]
```

the filter key is `doc-5501…-chunk-000`; `1` is only the display index.

Every returned rectangle has a stable range-based identifier:

```text
hl_<blockid>_<start>_<end>_<box-ordinal>
```

For example, `hl_287cc…_25_184_0` is the first PDF box for characters
`[25, 184)` in that block. When a claim cites several non-contiguous blocks or
ranges, the response returns several highlight objects—each carrying its own
`highlight_id` and the exact `citation_id` marker that selected it. Highlights
from other blocks on the same page are removed.

### Empty `sources` regression

An empty payload such as:

```json
{"citation_highlights": {"version": 1, "sources": {}}}
```

can result when highlights were built correctly but then filtered using `1`,
`2`, and so on instead of the stored chunk IDs. The EDW citation parser now
extracts the selector after `:` and before `§`, so response13-style citations
keep their matching highlight chunks.

## Viewer rules

1. Parse `[^Index:ChunkID§SectionID¶BlockID]` without discarding `§` or `¶`.
2. Resolve the cited block ID in `blocks.jsonl`.
3. Draw only that block's position boxes. Do not highlight all blocks on the
   page and do not highlight every block included in the embedding chunk.
4. If character offsets and a validated marked quote are available, limit the
   text-level presentation to that exact range. PDF bbox data is normally
   block-level unless the parser also emits word/glyph geometry.
5. If positions are missing, show a non-overlay citation fallback rather than
   guessing a page location.

## Known quirks and compatibility notes

- Existing records created before block provenance was stored must be reprocessed
  to gain reliable visual citations.
- A PDF can have duplicate block text. Position lookup must prefer block ID over
  text matching to avoid highlighting the wrong repeated paragraph.
- `V` chunking can normalize whitespace; provenance backfill accepts
  whitespace-equivalent source spans. This is safe for block selection but is
  not sufficient to prove an exact direct quote.
- A token boundary through a multi-byte character can produce the replacement
  character (`�`). Such a chunk deliberately receives no provenance rather than
  an unsafe guessed block.
- Re-indexing with a different parser, parser version, or normalized content can
  change block IDs. Cached answers with old `¶blockid` values should not be used
  against a newly parsed document without validation.
- `citation_highlights` currently stores block/page boxes per chunk. It is a
  compatibility layer for the existing EDW viewer; the canonical visual source
  remains `blocks.jsonl` plus the citation's `¶blockid`.

## Relevant implementation files

- `lightrag/sidecar/writer.py` — writes `blocks.jsonl` and block positions.
- `edw_rag_patches/__init__.py` — installs a runtime replacement for F/R/V
  sidecar backfill, query response models, and request serialization; it does
  not edit LightRAG source files.
- `edw_rag_patches/query_chain.py` — restores provenance after retrieval and
  includes it in structured query output without modifying `lightrag/operate.py`
  or `lightrag/utils.py`.
- `edw_rag_patches/sidecar.py` — constructs and filters `citation_highlights`.
- `edw_rag_patches/__init__.py` — applies the EDW runtime wrappers.
