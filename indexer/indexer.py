import os
import sqlite3
import time
import hashlib
import logging
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCAN_PATH       = Path(os.environ.get("SCAN_PATH", "/mnt/nas"))
SCAN_INTERVAL   = int(os.environ.get("SCAN_INTERVAL", 3600))
QDRANT_HOST     = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT     = int(os.environ.get("QDRANT_PORT", 6333))
COLLECTION      = os.environ.get("QDRANT_COLLECTION", "documents")
OLLAMA_HOST     = os.environ.get("OLLAMA_HOST", "172.17.0.1")
OLLAMA_PORT     = int(os.environ.get("OLLAMA_PORT", 11434))
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "nomic-embed-text")
OCR_LANG        = os.environ.get("OCR_LANG", "deu+eng")
CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", 600))
CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP", 80))
INDEXER_THREADS = int(os.environ.get("INDEXER_THREADS", 2))

STATE_DIR   = Path("/app/indexer_state")
DB_PATH     = STATE_DIR / "index.db"
TMP_DIR     = STATE_DIR / "tmp"
VECTOR_DIM  = 768
EMBED_URL   = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/embeddings"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def wait_for_services() -> None:
    qdrant_url = f"http://{QDRANT_HOST}:{QDRANT_PORT}/healthz"
    ollama_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    log.info("Waiting for Qdrant and Ollama to become available...")
    while True:
        qdrant_ok = False
        ollama_ok = False
        try:
            r = requests.get(qdrant_url, timeout=5)
            qdrant_ok = r.status_code == 200
        except Exception:
            pass
        try:
            r = requests.get(ollama_url, timeout=5)
            ollama_ok = r.status_code == 200
        except Exception:
            pass
        if qdrant_ok and ollama_ok:
            log.info("Qdrant and Ollama are ready.")
            return
        if not qdrant_ok:
            log.info("Waiting for Qdrant at %s ...", qdrant_url)
        if not ollama_ok:
            log.info("Waiting for Ollama at %s ...", ollama_url)
        time.sleep(5)


def ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(COLLECTION)
        log.info("Collection '%s' already exists.", COLLECTION)
    except Exception:
        log.info("Creating collection '%s' ...", COLLECTION)
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="folder",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name="path",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("Collection '%s' created with payload indexes.", COLLECTION)


def init_db() -> sqlite3.Connection:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS indexed_files (
            path       TEXT PRIMARY KEY,
            mtime      REAL NOT NULL,
            indexed_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    log.info("SQLite database ready at %s", DB_PATH)
    return conn


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def has_text_layer(pdf_path: Path) -> bool:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_check = min(3, len(pdf.pages))
            for i in range(pages_to_check):
                text = pdf.pages[i].extract_text() or ""
                if len(text.strip()) > 20:
                    return True
    except Exception as exc:
        log.warning("has_text_layer failed for %s: %s", pdf_path, exc)
    return False


def extract_text_native(pdf_path: Path) -> list[tuple[int, str]]:
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = text.strip()
                if len(text) >= 10:
                    results.append((i, text))
    except Exception as exc:
        log.error("extract_text_native failed for %s: %s", pdf_path, exc)
    return results


def extract_text_ocr(pdf_path: Path) -> list[tuple[int, str]]:
    results = []
    try:
        images = convert_from_path(
            str(pdf_path),
            dpi=200,
            output_folder=str(TMP_DIR),
            fmt="jpeg",
            thread_count=1,
            use_pdftocairo=True,
        )
        for i, image in enumerate(images, start=1):
            try:
                text = pytesseract.image_to_string(image, lang=OCR_LANG)
                text = text.strip()
                if len(text) >= 10:
                    results.append((i, text))
            except Exception as exc:
                log.warning("OCR failed for page %d of %s: %s", i, pdf_path, exc)
            finally:
                del image
        del images
    except Exception as exc:
        log.error("extract_text_ocr failed for %s: %s", pdf_path, exc)
    return results


# ---------------------------------------------------------------------------
# Chunking and embedding
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        if len(chunk.strip()) >= 20:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def make_point_id(abs_path: str, page: int, chunk_idx: int) -> int:
    key = f"{abs_path}::{page}::{chunk_idx}"
    return int(hashlib.md5(key.encode()).hexdigest()[:16], 16)


# ---------------------------------------------------------------------------
# Qdrant operations
# ---------------------------------------------------------------------------

def delete_existing_chunks(client: QdrantClient, abs_path: str) -> None:
    client.delete(
        collection_name=COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="path", match=MatchValue(value=abs_path))]
        ),
    )


# ---------------------------------------------------------------------------
# Per-file indexing
# ---------------------------------------------------------------------------

def index_pdf(
    pdf_path: Path,
    client: QdrantClient,
    db_conn: sqlite3.Connection,
    db_lock: threading.Lock,
) -> None:
    abs_path = str(pdf_path.resolve())
    try:
        mtime = pdf_path.stat().st_mtime
    except OSError as exc:
        log.error("Cannot stat %s: %s", pdf_path, exc)
        return

    # Check if already indexed with same mtime
    with db_lock:
        row = db_conn.execute(
            "SELECT mtime FROM indexed_files WHERE path=?", (abs_path,)
        ).fetchone()
    if row and abs(row[0] - mtime) < 0.001:
        return  # unchanged, skip silently

    log.info("Indexing: %s", abs_path)

    try:
        # Compute path metadata
        try:
            rel_path = str(pdf_path.relative_to(SCAN_PATH))
        except ValueError:
            rel_path = abs_path

        parts = Path(rel_path).parts
        filename = parts[-1]
        folder = parts[0] if len(parts) > 1 else ""
        subfolders = list(parts[1:-1])

        # Delete old chunks before re-indexing
        delete_existing_chunks(client, abs_path)

        # Extract text
        is_native = has_text_layer(pdf_path)
        if is_native:
            pages = extract_text_native(pdf_path)
            source = "native"
        else:
            pages = extract_text_ocr(pdf_path)
            source = "ocr"

        if not pages:
            log.warning("No text extracted from %s", abs_path)
            with db_lock:
                db_conn.execute(
                    "INSERT OR REPLACE INTO indexed_files (path, mtime, indexed_at) VALUES (?,?,?)",
                    (abs_path, mtime, time.time()),
                )
                db_conn.commit()
            return

        # Build points
        points: list[PointStruct] = []
        total_chunks = 0

        for page_num, text in pages:
            chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    embedding = get_embedding(chunk)
                except Exception as exc:
                    log.error("Embedding failed (page %d, chunk %d) in %s: %s",
                               page_num, chunk_idx, abs_path, exc)
                    continue

                point_id = make_point_id(abs_path, page_num, chunk_idx)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "path": abs_path,
                            "rel_path": rel_path,
                            "filename": filename,
                            "folder": folder,
                            "subfolders": subfolders,
                            "page": page_num,
                            "chunk": chunk_idx,
                            "text": chunk,
                            "source": source,
                            "mtime": mtime,
                        },
                    )
                )
                total_chunks += 1

                # Upsert in batches of 64
                if len(points) >= 64:
                    client.upsert(collection_name=COLLECTION, points=points)
                    points = []

        # Upsert remaining
        if points:
            client.upsert(collection_name=COLLECTION, points=points)

        # Mark as indexed in SQLite
        with db_lock:
            db_conn.execute(
                "INSERT OR REPLACE INTO indexed_files (path, mtime, indexed_at) VALUES (?,?,?)",
                (abs_path, mtime, time.time()),
            )
            db_conn.commit()

        log.info("Indexed %s: %d chunks, source=%s", filename, total_chunks, source)

    except Exception as exc:
        log.error("Failed to index %s: %s", abs_path, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Scan loop
# ---------------------------------------------------------------------------

def cleanup_deleted(
    client: QdrantClient,
    db_conn: sqlite3.Connection,
    db_lock: threading.Lock,
) -> None:
    with db_lock:
        rows = db_conn.execute("SELECT path FROM indexed_files").fetchall()
    deleted = [row[0] for row in rows if not Path(row[0]).exists()]
    for abs_path in deleted:
        log.info("File removed, purging from index: %s", abs_path)
        try:
            delete_existing_chunks(client, abs_path)
        except Exception as exc:
            log.error("Failed to delete chunks for %s: %s", abs_path, exc)
        with db_lock:
            db_conn.execute("DELETE FROM indexed_files WHERE path=?", (abs_path,))
            db_conn.commit()


def scan_and_index(
    client: QdrantClient,
    db_conn: sqlite3.Connection,
    db_lock: threading.Lock,
) -> None:
    log.info("Starting scan of %s ...", SCAN_PATH)
    # Collect PDFs, deduplicate by resolved path (handles case-insensitive FS)
    seen: set[Path] = set()
    pdf_paths: list[Path] = []
    for pattern in ("*.pdf", "*.PDF"):
        for p in SCAN_PATH.rglob(pattern):
            resolved = p.resolve()
            if resolved not in seen:
                seen.add(resolved)
                pdf_paths.append(p)

    log.info("Found %d PDF files.", len(pdf_paths))

    with ThreadPoolExecutor(max_workers=INDEXER_THREADS) as executor:
        futures = {
            executor.submit(index_pdf, p, client, db_conn, db_lock): p
            for p in pdf_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                log.error("Unhandled error for %s: %s", path, exc)

    cleanup_deleted(client, db_conn, db_lock)
    log.info("Scan complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    wait_for_services()

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)
    db_conn = init_db()
    db_lock = threading.Lock()

    while True:
        scan_and_index(client, db_conn, db_lock)
        log.info("Next scan in %d seconds.", SCAN_INTERVAL)
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
