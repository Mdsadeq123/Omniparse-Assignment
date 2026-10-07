from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import IngestBatchResponse, IngestResponse
from app.services.ingestion import ingest_file

router = APIRouter(prefix="/api", tags=["ingest"])


@router.post("/ingest", response_model=IngestBatchResponse)
async def ingest(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    results: list[IngestResponse] = []
    errors: list[str] = []

    for file in files:
        try:
            result = await ingest_file(file)
            results.append(IngestResponse(**result))
        except Exception as e:
            msg = str(e).strip() or repr(e)
            errors.append(f"{file.filename}: {msg}")

    if not results and errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    return IngestBatchResponse(results=results, errors=errors)
