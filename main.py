#!/usr/bin/env python3
"""
Azure AI Search Custom Skill & Docling Batch Tester
Handles production RAG chunking via FastAPI and local Dev batch testing.
Optimized for memory efficiency, avoiding leaks by managing memory manually and cleaning up.
"""

import asyncio
import base64
import gc
import io
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ====================== CONFIG & CONSTANTS ======================
ENV = os.getenv("ENV", "prod").lower()

DOCLING_URL = os.getenv("DOCLING_URL", "http://docling:5001").rstrip("/")
DOCLING_API_KEY = os.getenv("DOCLING_API_KEY", "")
WRAPPER_SECRET = os.getenv("WRAPPER_SECRET", "")
DOCLING_TIMEOUT = min(
    float(os.getenv("DOCLING_TIMEOUT", "220.0")), 220.0
)  # Azure skill max is 230s

CHUNK_TARGET = int(os.getenv("CHUNK_TARGET", "1000"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
FAST_OCR_PAGE_THRESHOLD = int(os.getenv("FAST_OCR_PAGE_THRESHOLD", "20"))

INPUT_DIR = Path("/data/docs/in")
OUTPUT_DIR = Path("/data/docs/out")
SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".html", ".md"}


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("docling-rag-pipeline")


# ====================== MODELS ======================
class AzureSkillRecord(BaseModel):
    recordId: str
    data: Dict[str, Any] = Field(default_factory=dict)


class AzureSkillRequest(BaseModel):
    values: List[AzureSkillRecord]


class AzureSkillResponse(BaseModel):
    values: List[Dict[str, Any]]


# ====================== CORE LOGIC ======================
def extract_file_bytes(file_data: Any) -> bytes:
    """Extract binary data from Azure AI Search's file_data payload."""
    if not file_data:
        raise ValueError("Missing file_data")
    if isinstance(file_data, str):
        return base64.b64decode(file_data)
    if isinstance(file_data, dict):
        for key in ("$content", "data", "content", "base64"):
            val = file_data.get(key)
            if isinstance(val, str):
                return base64.b64decode(val)
    raise ValueError(f"Unsupported file_data format: {type(file_data)}")


def chunk_markdown_with_langchain(
    markdown: str, metadata: dict
) -> List[Dict[str, str]]:
    """Chunk text for embedding. Metadata stored as separate index fields, not in chunk text."""
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_MAX,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n## ",  # markdown H2 headings
            "\n### ",  # markdown H3 headings
            "\n\n",  # paragraph breaks
            "\n|",  # table row boundary (new!)
            "\n",  # line breaks
            ". ",  # sentences
            " ",
            "",  # fallback
        ],
        length_function=len,
    )
    chunks = recursive_splitter.split_text(markdown)
    return [{"text": c} for c in chunks]


def get_pdf_page_count(file_bytes: bytes) -> int:
    """Get page count from PDF bytes without rendering. Returns 0 for non-PDFs."""
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return len(reader.pages)
    except Exception:
        return 0


def extract_text_with_pypdf(file_bytes: bytes, file_name: str) -> str:
    """Fast text extraction using pypdf. Always returns text (may be empty for scanned PDFs)."""
    try:
        t0 = time.monotonic()
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages_text.append(text.strip())
        elapsed = time.monotonic() - t0
        full_text = "\n\n".join(pages_text)
        logger.info(
            f"pypdf extraction | Chars: {len(full_text):,} | Pages: {len(reader.pages)} | Time: {elapsed:.1f}s | File: {file_name}"
        )
        return full_text
    except Exception as e:
        logger.error(f"pypdf extraction failed | File: {file_name} | Error: {e}")
        return ""


async def process_document_via_docling(
    client: httpx.AsyncClient,
    file_bytes: bytes,
    file_name: str,
    fast_mode: bool = False,
) -> str:
    """Send document to Docling. Uses fast OCR settings for large documents."""
    file_size_mb = len(file_bytes) / (1024 * 1024)
    logger.info(
        f"Preparing Docling request | File: {file_name} | Size: {file_size_mb:.2f} MB | Fast: {fast_mode}"
    )

    b64_string = base64.b64encode(file_bytes).decode("utf-8")

    if fast_mode:
        options = {
            "do_ocr": True,
            "ocr_engine": "rapidocr",
            "ocr_lang": ["en", "de"],
            "do_table_structure": True,
            "table_mode": "fast",
            "image_export_mode": "placeholder",
            "to_formats": ["md"],
        }
    else:
        options = {
            "do_ocr": True,
            "ocr_lang": ["en", "de"],
            "do_table_structure": True,
            "table_mode": "accurate",
            "image_export_mode": "placeholder",
            "to_formats": ["md"],
        }

    payload = {
        "options": options,
        "target": {"kind": "inbody"},
        "sources": [
            {"kind": "file", "base64_string": b64_string, "filename": file_name}
        ],
    }

    # Free memory immediately before awaiting HTTP request
    del file_bytes
    del b64_string

    headers = {"Content-Type": "application/json"}
    if DOCLING_API_KEY:
        headers["X-Api-Key"] = DOCLING_API_KEY

    t0 = time.monotonic()
    logger.info(
        f"Sending to Docling | URL: {DOCLING_URL}/v1/convert/source | File: {file_name}"
    )

    try:
        response = await client.post(
            f"{DOCLING_URL}/v1/convert/source",
            json=payload,
            headers=headers,
        )
    except httpx.ConnectError as e:
        logger.error(f"Docling connection failed | URL: {DOCLING_URL} | Error: {e}")
        raise
    except httpx.TimeoutException as e:
        elapsed = time.monotonic() - t0
        logger.error(
            f"Docling timeout after {elapsed:.1f}s | File: {file_name} | Error: {e}"
        )
        raise

    elapsed = time.monotonic() - t0
    logger.info(
        f"Docling responded | Status: {response.status_code} | Time: {elapsed:.1f}s | File: {file_name}"
    )

    response.raise_for_status()
    result_json = response.json()

    # Cleanup response payload string to save memory
    del payload

    # Extract Markdown
    doc = result_json.get("document", {})
    markdown = (
        doc.get("md_content")
        or doc.get("markdown_content")
        or doc.get("markdown")
        or ""
    ).strip()
    logger.info(f"Extracted markdown | Chars: {len(markdown):,} | File: {file_name}")

    # Final cleanup
    del result_json
    del doc

    return markdown


# ====================== FASTAPI FOR PROD (AZURE) ======================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize persistent client for connection pooling
    app.state.http_client = httpx.AsyncClient(timeout=DOCLING_TIMEOUT)
    yield
    await app.state.http_client.aclose()


app = FastAPI(title="Azure AI Search Docling Wrapper", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "env": ENV}


@app.post("/azure-search/docling", response_model=AzureSkillResponse)
async def azure_search_docling(
    request: AzureSkillRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    # Log incoming request details
    api_key_status = "present" if x_api_key else "missing"
    api_key_match = (
        "valid"
        if (WRAPPER_SECRET and x_api_key == WRAPPER_SECRET)
        else ("no-secret-configured" if not WRAPPER_SECRET else "invalid")
    )
    record_count = len(request.values)
    record_summaries = []
    for r in request.values:
        data_keys = list(r.data.keys())
        file_data_size = None
        fd = r.data.get("file_data")
        if isinstance(fd, str):
            file_data_size = f"{len(fd) / 1024:.0f}KB"
        elif isinstance(fd, dict):
            content = fd.get("$content", fd.get("data", fd.get("content", "")))
            if isinstance(content, str):
                file_data_size = f"{len(content) / 1024:.0f}KB"
        record_summaries.append(
            f"id={r.recordId} keys={data_keys} file_data={file_data_size or 'N/A'}"
        )
    logger.info(
        f"REQUEST IN | Records: {record_count} | API-Key: {api_key_status}/{api_key_match} | {'; '.join(record_summaries)}"
    )

    if ENV == "dev":
        logger.warning("Received API request while in DEV mode.")

    if WRAPPER_SECRET and x_api_key != WRAPPER_SECRET:
        logger.warning(
            f"AUTH REJECTED | API-Key: {api_key_status} | Match: {api_key_match}"
        )
        raise HTTPException(status_code=401, detail="Unauthorized")

    client: httpx.AsyncClient = app.state.http_client

    async def process_record(record: AzureSkillRecord) -> Dict[str, Any]:
        record_id = record.recordId
        data = record.data
        try:
            file_name = data.get("file_name", "document.bin")
            path = data.get("path", "")

            logger.info(f"Processing recordId: {record_id} | File: {file_name}")

            file_bytes = extract_file_bytes(data.get("file_data"))
            data.pop("file_data", None)

            fast_mode = False
            if file_name.lower().endswith(".pdf"):
                page_count = get_pdf_page_count(file_bytes)
                fast_mode = page_count > FAST_OCR_PAGE_THRESHOLD
                logger.info(f"PDF pages: {page_count} | Fast mode: {fast_mode}")

            if fast_mode:
                markdown = extract_text_with_pypdf(file_bytes, file_name)
            else:
                markdown = await process_document_via_docling(
                    client, file_bytes, file_name, fast_mode=False
                )

            metadata = {"file_name": file_name, "path": path}
            chunks = chunk_markdown_with_langchain(markdown, metadata)
            del markdown

            return {
                "recordId": record_id,
                "data": {"chunks": chunks},
                "errors": [],
                "warnings": [] if chunks else [{"message": "No chunks generated."}],
            }
        except Exception as e:
            logger.error(f"Failed recordId={record_id}: {str(e)}")
            return {
                "recordId": record_id,
                "data": {"chunks": []},
                "errors": [{"message": str(e)}],
                "warnings": [],
            }
        finally:
            gc.collect()

    try:
        results = await asyncio.gather(*[process_record(r) for r in request.values])
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        results = [
            {
                "recordId": r.recordId,
                "data": {"chunks": []},
                "errors": [{"message": f"Batch error: {str(e)}"}],
                "warnings": None,
            }
            for r in request.values
        ]
    # Log response summary
    response_values = list(results)
    for rv in response_values:
        rid = rv.get("recordId", "?")
        chunks = rv.get("data", {}).get("chunks", [])
        errors = rv.get("errors") or []
        warnings = rv.get("warnings") or []
        logger.info(
            f"RESPONSE OUT | recordId={rid} | chunks={len(chunks)} | errors={len(errors)} | warnings={len(warnings)}"
        )
        if errors:
            logger.warning(
                f"RESPONSE ERRORS | recordId={rid} | {[e.get('message', '') for e in errors]}"
            )

    return AzureSkillResponse(values=response_values)


# ====================== DEV MODE BATCH TEST ======================
async def dev_batch_test():
    """Runs the directory batch test without FastAPI."""
    logger.info("=== Docling Batch Test Mode (DEV) ===")
    logger.info(f"Input dir : {INPUT_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Docling URL: {DOCLING_URL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [
        f
        for f in INPUT_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    ]
    if not files:
        logger.warning(f"No supported files found in {INPUT_DIR}")
        return

    logger.info(f"Found {len(files)} files. Starting processing...")
    success_count = 0

    async with httpx.AsyncClient(timeout=DOCLING_TIMEOUT) as client:
        for file_path in sorted(files):
            output_md = OUTPUT_DIR / f"{file_path.stem}.md"
            if output_md.exists():
                logger.info(f"Overwriting: {file_path.name}")

            try:
                with open(file_path, "rb") as f:
                    file_bytes = f.read()

                # Use fast mode for large PDFs
                fast_mode = False
                if file_path.suffix.lower() == ".pdf":
                    page_count = get_pdf_page_count(file_bytes)
                    fast_mode = page_count > FAST_OCR_PAGE_THRESHOLD
                    logger.info(f"PDF pages: {page_count} | Fast mode: {fast_mode}")

                if fast_mode:
                    markdown = extract_text_with_pypdf(file_bytes, file_path.name)
                else:
                    markdown = await process_document_via_docling(
                        client, file_bytes, file_path.name, fast_mode=False
                    )

                # Save Raw Markdown
                output_md.write_text(markdown, encoding="utf-8")

                # Create sample chunks and dump for Dev review
                metadata = {
                    "file_name": file_path.name,
                    "path": f"/data/docs/in/{file_path.name}",
                }
                chunks = chunk_markdown_with_langchain(markdown, metadata)

                chunks_json_path = OUTPUT_DIR / f"{file_path.stem}.chunks.json"
                chunks_json_path.write_text(
                    json.dumps(chunks, indent=2), encoding="utf-8"
                )

                logger.info(
                    f"✓ Saved: {output_md.name} ({len(markdown):,} chars) -> {len(chunks)} chunks"
                )
                success_count += 1

            except Exception as e:
                logger.error(f"✗ Failed {file_path.name}: {e}")
            finally:
                gc.collect()

    logger.info(f"=== Batch finished === {success_count}/{len(files)} files succeeded")


# ====================== ENTRY POINT ======================
if __name__ == "__main__":
    if ENV == "dev":
        import asyncio

        asyncio.run(dev_batch_test())
    else:
        import uvicorn

        logger.info("Starting FastAPI Production Server...")
        uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=1, log_level="info")
