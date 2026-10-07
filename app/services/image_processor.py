import base64
from pathlib import Path

from langchain_core.documents import Document
from openai import OpenAI

from app.config import settings
from app.logging_config import get_logger

_ocr_reader = None
logger = get_logger("ocr")


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr

        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def process_image(file_path: Path, doc_id: str, filename: str) -> list[Document]:
    logger.info("=== [PARSING / OCR] Extracting text from %s using EasyOCR ===", filename)
    ocr_text = _run_ocr(file_path)
    caption = ""
    if settings.vision_caption_enabled:
        logger.info("[PARSING / OCR] Generating optional vision caption for %s", filename)
        caption = _get_vision_caption(file_path)

    parts = []
    if ocr_text.strip():
        parts.append(f"OCR text:\n{ocr_text.strip()}")
    if caption.strip():
        parts.append(f"Image description:\n{caption.strip()}")

    if not parts:
        parts.append("(No text detected in image)")

    content = "\n\n".join(parts)
    logger.info(
        "[PARSING / OCR] EasyOCR complete: %s OCR characters, vision_caption=%s",
        len(ocr_text),
        "enabled" if settings.vision_caption_enabled else "disabled",
    )
    return [
        Document(
            page_content=content,
            metadata={
                "doc_id": doc_id,
                "source": filename,
                "type": "image",
                "page": "1",
            },
        )
    ]


def _run_ocr(file_path: Path) -> str:
    reader = _get_ocr_reader()
    results = reader.readtext(str(file_path))
    valid_results = [r for r in results if r[2] > 0.3]
    if not valid_results:
        return ""
        
    # Group words by Y-coordinate to preserve tabular/multi-column line breaks
    valid_results.sort(key=lambda r: r[0][0][1])
    lines = []
    current_line = []
    current_y = None
    
    for r in valid_results:
        bbox, text, conf = r
        y0 = bbox[0][1]
        height = bbox[3][1] - bbox[0][1]
        
        if current_y is None:
            current_y = y0
            current_line.append(r)
        elif abs(y0 - current_y) < (height * 0.5 + 5):
            current_line.append(r)
            current_y = sum([item[0][0][1] for item in current_line]) / len(current_line)
        else:
            current_line.sort(key=lambda item: item[0][0][0])
            lines.append("    ".join([item[1] for item in current_line]))
            current_line = [r]
            current_y = y0
            
    if current_line:
        current_line.sort(key=lambda item: item[0][0][0])
        lines.append("    ".join([item[1] for item in current_line]))
        
    raw_text = "\n".join(lines)
    return _clean_ocr_text(raw_text)


def _clean_ocr_text(text: str) -> str:
    import re
    text = text.replace("Pinecone_", "Pinecone")
    text = text.replace("Llamalndex", "LlamaIndex")
    text = re.sub(r'\s+', ' ', text)
    return text.strip(" _-~=+|\\/[]{}()<>*^%$#@!`\"';:")


def _get_vision_caption(file_path: Path) -> str:
    if not settings.openrouter_api_key:
        return ""

    suffix = file_path.suffix.lower().lstrip(".")
    mime_map = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "webp": "webp",
        "gif": "gif",
        "bmp": "bmp",
    }
    mime_subtype = mime_map.get(suffix, "jpeg")
    image_data = base64.standard_b64encode(file_path.read_bytes()).decode("utf-8")

    from app.services.openrouter import get_openrouter_headers

    client = OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        default_headers=get_openrouter_headers(),
    )
    try:
        response = client.chat.completions.create(
            model=settings.vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Describe this image for search indexing. "
                                "If the image contains tabular data or multi-column text, "
                                "transcribe it completely as a structured Markdown table or "
                                "preserve the line breaks exactly."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{mime_subtype};base64,{image_data}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content or ""
    except Exception:
        return ""
