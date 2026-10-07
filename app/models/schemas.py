from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_added: int
    status: str


class IngestBatchResponse(BaseModel):
    results: list[IngestResponse]
    errors: list[str] = []


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    top_k: int | None = None


class SourceCitation(BaseModel):
    file: str
    snippet: str
    score: float | None = None
    page: str | None = None
    doc_type: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    doc_type: str
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class DeleteResponse(BaseModel):
    doc_id: str
    deleted: bool
    message: str
