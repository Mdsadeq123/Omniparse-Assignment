
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Your existing imports continue below:
from fastapi import FastAPI
from app.api import chat, documents, ingest
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api import chat, documents, ingest
from app.config import BASE_DIR, settings

app = FastAPI(
    title="OmniParse",
    description="Multimedia RAG summarization engine — ingest documents and images, query with grounded AI.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(chat.router)
app.include_router(documents.router)

static_dir = BASE_DIR / "static"
settings.uploads_path.mkdir(parents=True, exist_ok=True)
settings.chroma_path.mkdir(parents=True, exist_ok=True)

if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/health")
async def health():
    return {"status": "ok", "api_key_set": bool(settings.openrouter_api_key)}


@app.get("/")
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return FileResponse(index)
    return {"message": "OmniParse API", "docs": "/docs"}
