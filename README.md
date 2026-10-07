# OmniParse - Multimodal RAG Engine

OmniParse is a FastAPI-based multimodal Retrieval-Augmented Generation (RAG) application. It accepts documents and images, indexes their contents in a local ChromaDB vector store, and answers questions with NVIDIA Nemotron models through OpenRouter.

The browser interface displays each answer with the filename, page number, similarity score, and expandable source snippets used to generate it.

## Features

- PDF extraction with PyPDF.
- DOCX, TXT, Markdown, and image ingestion.
- Image OCR with EasyOCR for PNG, JPG, JPEG, WEBP, GIF, and BMP files.
- Optional vision-language captions for image indexing.
- Recursive text chunking with overlap.
- Batched OpenRouter embeddings and persistent ChromaDB storage.
- Grounded RAG answers with page-aware citations and source drawers.
- Duplicate upload detection using SHA-256 file hashes.
- Document listing and deletion in the browser interface.
- Timestamped ingestion logs for loading, parsing/OCR, chunking, embedding, and ChromaDB writes.

## Architecture

```text
Upload file
  -> PDF: PyPDF extraction
  -> DOCX/TXT/Markdown: text extraction
  -> Image: EasyOCR (+ optional vision caption)
  -> Recursive chunking (800 characters, 100-character overlap)
  -> OpenRouter embeddings (batched)
  -> Persistent ChromaDB collection
  -> Top-k similarity retrieval
  -> Nemotron-powered answer + source citations
```

### Workflow

1. The upload endpoint validates the file, calculates a SHA-256 hash, and stores the upload locally.
2. PyPDF extracts PDF text, supported text formats are read directly, and EasyOCR extracts image text.
3. Extracted content is split into overlapping chunks. Every chunk retains its filename, document ID, type, and page metadata.
4. The chunks are embedded in batches and stored in the persistent ChromaDB collection.
5. For a question, OmniParse retrieves the most relevant chunks, sends them as context to the chat model, and returns the answer with its source snippets.
6. The prompt instructs the model to answer only from retrieved context and decline unrelated questions.

## Tech Stack

| Area | Tools |
| --- | --- |
| API server | FastAPI, Uvicorn |
| Browser UI | HTML, CSS, vanilla JavaScript |
| RAG orchestration | LangChain |
| Vector database | ChromaDB |
| PDF processing | PyPDF |
| OCR | EasyOCR |
| LLM and embeddings | OpenRouter with NVIDIA Nemotron models |
| Configuration | Pydantic Settings, python-dotenv |

### Version Bounds

| Component | Supported version |
| --- | --- |
| Python | 3.11+ |
| FastAPI | `>=0.115,<1.0` |
| ChromaDB | `>=0.5,<2.0` |
| LangChain | `>=0.3,<2.0` |
| EasyOCR | `>=1.7,<2.0` |
| PyPDF | `>=5.0,<7.0` |

## Requirements

- Python 3.11 or newer
- An OpenRouter API key
- Git
- Optional: `uv` for faster dependency installation

## Local Setup

### 1. Clone the project

```bash
git clone https://github.com/Mdsadeq123/Omniparse-Assignment.git
cd Omniparse-Assignment
```

### 2. Create a virtual environment

```bash
uv venv --python 3.11 .venv
```

Or:

```bash
python -m venv .venv
```

### 3. Install dependencies

Windows with `uv`:

```powershell
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
```

macOS or Linux with `uv`:

```bash
uv pip install --python .venv/bin/python -r requirements.txt
```

Alternatively, activate the environment and run `python -m pip install -r requirements.txt`.

### 4. Configure environment variables

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Set `OPENROUTER_API_KEY` in `.env`. Never commit a real key or the `.env` file.

### 5. Start OmniParse

```bash
python run.py
```

On Windows without activating the environment:

```powershell
.\.venv\Scripts\python.exe run.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Using and Testing the Application

1. Start the server and keep the terminal visible.
2. Upload a supported document or image.
3. Watch for `[DOCUMENT LOADING]`, `[PARSING / OCR]`, `[CHUNKING]`, and `[EMBEDDING & VECTOR DB]` logs.
4. Confirm the document and its chunk count appear in the left-side document list.
5. Ask a question and expand the source drawer beneath the answer to inspect the filename, page, score, and supporting snippet.
6. Ask an unrelated question to verify the no-relevant-information response.
7. Use **Delete** to remove a document and its local upload.

The health endpoint is available at [http://localhost:8000/health](http://localhost:8000/health).

## Chunking Strategy

OmniParse uses LangChain's **Recursive Character Text Splitter** with an 800-character chunk size and a 100-character overlap. The recursive strategy prefers natural boundaries such as paragraphs and lines before using smaller boundaries. This preserves semantic context at paragraph boundaries while avoiding abrupt truncation in complex tables, OCR output, and multi-line document structures. The overlap keeps nearby context available when an idea spans two chunks.

## Vector DB & Embedding Model Choice

**Vector database: ChromaDB.** ChromaDB provides lightweight, persistent local disk storage without an external cloud dependency for vector persistence. It also integrates directly with LangChain and supports metadata-backed similarity retrieval for filenames, document IDs, source types, and page numbers.

**Embedding model: OpenRouter with NVIDIA Nemotron embeddings.** The application sends embeddings in batches through OpenRouter using NVIDIA Nemotron embedding models. This choice provides strong semantic matching across mixed-domain text and the extracted text produced by the multimodal pipeline, including PDF and OCR content.

## Example Queries & Answer Themes

The committed sample corpus in `docs/` provides material for the following test queries:

1. **"What are the key findings or summary in Sadeq_Majid_Results.pdf?"** Expected theme: personality scores and the balance between practical and philosophical orientation.
2. **"What is the recommended 7-day meal plan strategy for mothers?"** Expected theme: a daily high-protein dietary breakdown and repeatable meal-planning structure.
3. **"Summarize the extracted text from the uploaded JPEG image."** Expected theme: OCR-extracted textual context from an indexed image.
4. **"How does the system handle document chunking and metadata tracking?"** Expected theme: document IDs, page numbers, source metadata, and chunk counts.
5. **"What is the capital of France?"** Expected theme: the out-of-scope refusal response, `No relevant information found in your documents.`

## Configuration

| Variable | Description |
| --- | --- |
| `OPENROUTER_API_KEY` | Required key for OpenRouter chat and embeddings. |
| `CHAT_MODEL` | Model used to generate answers. |
| `EMBEDDING_MODEL` | Model used to embed files and questions. |
| `VISION_MODEL` | Model used for optional image captions. |
| `CHROMA_PERSIST_DIR` | Local ChromaDB storage directory. |
| `UPLOAD_DIR` | Local directory for uploaded files. |
| `TOP_K` | Number of chunks retrieved for each question. |
| `MAX_RETRIEVAL_DISTANCE` | Optional similarity-distance filter; `0` disables it. |
| `MAX_UPLOAD_MB` | Maximum upload size in megabytes. |
| `VISION_CAPTION_ENABLED` | Enables optional image descriptions when `true`. |

## Project Structure

- `app/api/`: upload, chat, and document-management API routes.
- `app/models/`: Pydantic request and response schemas.
- `app/services/`: parsing, OCR, chunking, embeddings, ChromaDB, and RAG logic.
- `static/`: browser interface assets.
- `data/chroma/`: local ChromaDB persistence.
- `data/uploads/`: locally stored uploads.
- `run.py`: application entry point.

## Sample Documents

The `docs/` directory contains five original, non-trivial sample documents for local indexing and demonstration. The corpus includes a PDF and four text documents, each containing at least 500 words of content. These files are committed to the repository and are not runtime uploads.

## Limitations

- OCR accuracy depends on image quality and layout.
- Retrieval quality depends on uploaded content, chunking settings, embedding model, and `TOP_K`.
- Chat and embeddings require network access to OpenRouter.
- Model availability and rate limits can affect response time.
- Source snippets help verify answers but should not be the only basis for high-stakes decisions.

## Security

Store secrets only in `.env`. The repository ignores `.env`, virtual environments, ChromaDB data, uploaded files, and generated logs.
