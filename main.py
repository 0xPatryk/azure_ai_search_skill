#!/usr/bin/env python3
"""
Azure AI Search Custom Skill & Docling Batch Tester
Handles production RAG chunking via FastAPI and local Dev batch testing.
Optimized for memory efficiency, avoiding leaks by managing memory manually and cleaning up.
"""

import base64
import gc
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# ====================== CONFIG & CONSTANTS ======================
ENV = os.getenv("ENV", "prod").lower()

DOCLING_URL = os.getenv("DOCLING_URL", "http://docling:5001").rstrip("/")
DOCLING_API_KEY = os.getenv("DOCLING_API_KEY", "")
WRAPPER_SECRET = os.getenv("WRAPPER_SECRET", "")
DOCLING_TIMEOUT = float(os.getenv("DOCLING_TIMEOUT", "600.0"))

CHUNK_TARGET = int(os.getenv("CHUNK_TARGET", "1000"))
CHUNK_MAX = int(os.getenv("CHUNK_MAX", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

INPUT_DIR = Path("/data/docs/in")
OUTPUT_DIR = Path("/data/docs/out")
SUPPORTED_EXT = {".pdf", ".docx", ".xlsx", ".pptx", ".txt", ".html", ".md"}



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
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
        for key in ("data", "$content", "content", "base64"):
            val = file_data.get(key)
            if isinstance(val, str):
                return base64.b64decode(val)
    raise ValueError(f"Unsupported file_data format: {type(file_data)}")


def chunk_markdown_with_langchain(markdown: str, metadata: dict) -> List[str]:
    """Basic chunking enriched with source metadata."""
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_MAX,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    final_chunks = []
    file_name = metadata.get("file_name", "Unknown")
    path = metadata.get("path", "")
    chunk_prefix = f"---\nDocument: {file_name}\nPath: {path}\n---\n"
    
    sub_chunks = recursive_splitter.split_text(markdown)
    for sub in sub_chunks:
        final_chunks.append(f"{chunk_prefix}{sub}".strip())
    
    return [c for c in final_chunks if len(c) > 100]


async def process_document_via_docling(client: httpx.AsyncClient, file_bytes: bytes, file_name: str) -> str:
    """Send document to Docling in memory, extract markdown and image descriptions."""
    b64_string = base64.b64encode(file_bytes).decode("utf-8")
    
    # Docling accuracy best practices payload
    payload = {
        "options": {
            "do_ocr": True,
            "ocr_lang": ["en", "de"],
            "do_table_structure": True,
            "table_mode": "accurate",
            "image_export_mode": "placeholder",
            "to_formats": ["md"]
        },
        "target": {"kind": "inbody"},
        "sources": [
            {
                "kind": "file",
                "base64_string": b64_string,
                "filename": file_name
            }
        ]
    }
    
    # Free memory immediately before awaiting HTTP request
    del file_bytes
    del b64_string
    
    headers = {"Content-Type": "application/json"}
    if DOCLING_API_KEY:
        headers["X-Api-Key"] = DOCLING_API_KEY

    response = await client.post(
        f"{DOCLING_URL}/v1/convert/source",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    result_json = response.json()
    
    # Cleanup response payload string to save memory
    del payload 

    # Extract Markdown
    doc = result_json.get("document", {})
    markdown = (doc.get("md_content") or doc.get("markdown_content") or doc.get("markdown") or "").strip()

    # Final cleanup
    del result_json
    del doc
    
    return markdown if markdown else f"# Empty or unprocessable document\n\nFile: {file_name}"

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
    x_api_key: Optional[str] = Header(None, alias="x-api-key")
):
    if ENV == "dev":
        logger.warning("Received API request while in DEV mode.")
        
    if WRAPPER_SECRET and x_api_key != WRAPPER_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

    client: httpx.AsyncClient = app.state.http_client
    results = []

    for record in request.values:
        record_id = record.recordId
        data = record.data
        
        try:
            file_name = data.get("file_name", "document.bin")
            path = data.get("path", "")
            
            logger.info(f"Processing recordId: {record_id} | File: {file_name}")
            
            file_bytes = extract_file_bytes(data.get("file_data"))
            
            # Remove file_data from memory immediately
            data.pop("file_data", None)
            
            markdown = await process_document_via_docling(client, file_bytes, file_name)
            
            metadata = {"file_name": file_name, "path": path}
            chunks = chunk_markdown_with_langchain(markdown, metadata)
            
            # Clear markdown from memory
            del markdown
            
            results.append({
                "recordId": record_id,
                "data": {"chunks": chunks},
                "errors": None,
                "warnings": None if chunks else [{"message": "No chunks generated."}]
            })
            
        except Exception as e:
            logger.error(f"Failed recordId={record_id}: {str(e)}")
            results.append({
                "recordId": record_id,
                "data": {"chunks": []},
                "errors": [{"message": str(e)}],
                "warnings": None
            })
        finally:
            # Force garbage collection per large document to prevent Azure Batch OOM
            gc.collect()

    return AzureSkillResponse(values=results)


# ====================== DEV MODE BATCH TEST ======================
async def dev_batch_test():
    """Runs the directory batch test without FastAPI."""
    logger.info("=== Docling Batch Test Mode (DEV) ===")
    logger.info(f"Input dir : {INPUT_DIR}")
    logger.info(f"Output dir: {OUTPUT_DIR}")
    logger.info(f"Docling URL: {DOCLING_URL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    files = [f for f in INPUT_DIR.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
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

                markdown = await process_document_via_docling(client, file_bytes, file_path.name)
                
                # Save Raw Markdown
                output_md.write_text(markdown, encoding="utf-8")
                
                # Create sample chunks and dump for Dev review
                metadata = {"file_name": file_path.name, "path": f"/data/docs/in/{file_path.name}"}
                chunks = chunk_markdown_with_langchain(markdown, metadata)
                
                chunks_json_path = OUTPUT_DIR / f"{file_path.stem}.chunks.json"
                chunks_json_path.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

                logger.info(f"✓ Saved: {output_md.name} ({len(markdown):,} chars) -> {len(chunks)} chunks")
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