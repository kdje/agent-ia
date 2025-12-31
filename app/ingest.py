import os, json, math, uuid, zipfile, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import urllib.request, urllib.error
import xml.etree.ElementTree as ET

from docx import Document
from docx.opc.exceptions import PackageNotFoundError
from pptx import Presentation
import openpyxl
import ollama

# ----------------- Config -----------------
COLLECTION = os.getenv("QDRANT_COLLECTION", "integration_kb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")  # ex: ollama pull bge-m3 (ou autre embed model)

INCLUDE_EXT = {".md", ".txt", ".docx", ".pptx", ".xlsx", ".xmind"}

# Exclusions (optionnel)
EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", "node_modules", "howto", "setup"}
EXCLUDE_FILES = {
    "integration-architect-copilot_steps_switch-models.md",
}

# Chunking
CHUNK_SIZE = 1100
OVERLAP = 180
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # batch embeddings + upsert

# Dédup optionnelle (évite les doublons si tu relances ingest.py)
# 0 = ids aléatoires (comportement actuel), 1 = ids déterministes par hash (recommandé)
DEDUPE = os.getenv("DEDUPE", "0") == "1"

# ----------------- Utils -----------------
def l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return vec if n == 0 else [x / n for x in vec]

def _sanitize_text(s: str) -> str:
    # supprime les NUL + caractères de contrôle fréquents
    s = s.replace("\x00", " ")
    s = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", " ", s)
    # normalise unicode "safe"
    s = s.encode("utf-8", "ignore").decode("utf-8", "ignore")
    return s.strip()

def _is_finite_vec(v: List[float]) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)

def embed_texts(texts: List[str], normalize: bool = True) -> List[Optional[List[float]]]:
    """
    Embeddings via Ollama (local) robustes :
    - tente en batch
    - si erreur (ex: NaN => 500) => retry 1 par 1
    - si un chunk échoue => retourne None pour ce chunk
    """
    texts = [_sanitize_text(t) for t in texts]

    def _post(v: List[float]) -> Optional[List[float]]:
        if not _is_finite_vec(v):
            return None
        return l2_normalize(v) if normalize else v

    # 1) Essai batch
    try:
        resp = ollama.embed(model=EMBED_MODEL, input=texts)
        vecs = resp["embeddings"]
        return [_post(v) for v in vecs]
    except Exception as e:
        print(f"⚠️ embed batch failed ({type(e).__name__}): {e} -> retry 1-by-1")

    # 2) Fallback 1 par 1
    out: List[Optional[List[float]]] = []
    for t in texts:
        try:
            r = ollama.embed(model=EMBED_MODEL, input=t)
            v = r["embeddings"][0]
            vv = _post(v)
            if vv is None:
                preview = (t[:160] + "…") if len(t) > 160 else t
                print(f"⚠️ embed skip (non-finite vec) | text='{preview}'")
            out.append(vv)
        except Exception as e2:
            preview = (t[:160] + "…") if len(t) > 160 else t
            print(f"⚠️ embed skip ({type(e2).__name__}): {e2} | text='{preview}'")
            out.append(None)
    return out

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    text = text.replace("\r\n", "\n")
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + chunk_size])
        i += (chunk_size - overlap)
    return [c.strip() for c in chunks if c.strip()]

def should_skip(path: Path) -> bool:
    if path.name in EXCLUDE_FILES:
        return True
    for parent in path.parents:
        if parent.name in EXCLUDE_DIRS:
            return True
    return False

def _skip(part_reason: str, path: Path, **extra) -> List[Tuple[str, Dict[str, Any]]]:
    meta = {"part": "skip", "reason": part_reason, "source": str(path)}
    meta.update(extra)
    return [("", meta)]

def make_point_id(source_rel: str, part: str, chunk_text: str) -> str:
    """
    ID déterministe (évite doublons si relance). Qdrant accepte string IDs.
    """
    key = f"{source_rel}|{part}|{chunk_text}"
    return hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()

def get_embedding_dim() -> int:
    v = embed_texts(["dim-check"], normalize=True)[0]
    if v is None:
        # retry ultra simple
        v = embed_texts(["test"], normalize=True)[0]
    if v is None:
        raise RuntimeError(
            f"Impossible d'obtenir une dimension d'embedding via Ollama (model={EMBED_MODEL}). "
            f"Essaie un autre modèle d'embeddings (ex: nomic-embed-text / mxbai-embed-large) "
            f"ou réduis BATCH_SIZE."
        )
    return len(v)

# ----------------- Extractors -----------------
def read_txt_md(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    try:
        t = path.read_text(encoding="utf-8", errors="ignore")
        return [(t, {"part": "full", "source": str(path)})]
    except Exception as e:
        return _skip("txt_read_error", path, error=repr(e))

def read_docx(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        return _skip("missing_file", path)

    # DOCX must be a zip (OpenXML)
    if not zipfile.is_zipfile(str(path)):
        return _skip("not_a_docx_zip", path)

    try:
        doc = Document(str(path))
    except PackageNotFoundError as e:
        return _skip("docx_package_not_found", path, error=str(e))
    except Exception as e:
        return _skip("docx_open_error", path, error=repr(e))

    parts: List[Tuple[str, Dict[str, Any]]] = []
    base_meta = {"source": str(path)}

    # Paragraphs
    try:
        paras = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        if paras:
            parts.append(("\n".join(paras), {**base_meta, "part": "paragraphs"}))
    except Exception as e:
        parts.append(("", {**base_meta, "part": "skip", "reason": "docx_paragraphs_error", "error": repr(e)}))

    # Tables
    try:
        table_lines = []
        for ti, table in enumerate(doc.tables, start=1):
            for ri, row in enumerate(table.rows, start=1):
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    table_lines.append(f"[table {ti} row {ri}] " + " | ".join(cells))
        if table_lines:
            parts.append(("\n".join(table_lines), {**base_meta, "part": "tables"}))
    except Exception as e:
        parts.append(("", {**base_meta, "part": "skip", "reason": "docx_tables_error", "error": repr(e)}))

    return parts if any(t.strip() for t, _ in parts) else [("", {**base_meta, "part": "empty"})]

def read_pptx(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        return _skip("missing_file", path)

    try:
        prs = Presentation(str(path))
    except Exception as e:
        return _skip("pptx_open_error", path, error=repr(e))

    out: List[Tuple[str, Dict[str, Any]]] = []
    base_meta = {"source": str(path)}

    for idx, slide in enumerate(prs.slides, start=1):
        texts = []
        try:
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    s = shape.text.strip()
                    if s:
                        texts.append(s)
        except Exception as e:
            out.append(("", {**base_meta, "part": f"slide_{idx}", "reason": "pptx_shapes_error", "error": repr(e)}))
            continue

        # Notes (si présentes)
        try:
            if slide.has_notes_slide and slide.notes_slide:
                nt = slide.notes_slide.notes_text_frame.text.strip()
                if nt:
                    texts.append("[notes] " + nt)
        except Exception:
            pass

        if texts:
            out.append(("\n".join(texts), {**base_meta, "part": f"slide_{idx}"}))

    return out if out else [("", {**base_meta, "part": "empty"})]

def read_xlsx(path: Path, max_cells: int = 20000) -> List[Tuple[str, Dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        return _skip("missing_file", path)

    try:
        wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    except Exception as e:
        return _skip("xlsx_open_error", path, error=repr(e))

    out: List[Tuple[str, Dict[str, Any]]] = []
    base_meta = {"source": str(path)}

    try:
        for sheet in wb.worksheets:
            lines = []
            cell_count = 0
            for row in sheet.iter_rows():
                row_vals = []
                for cell in row:
                    if cell.value is None:
                        continue
                    v = str(cell.value).strip()
                    if v:
                        row_vals.append(v)
                        cell_count += 1
                    if cell_count >= max_cells:
                        break
                if row_vals:
                    lines.append(" | ".join(row_vals))
                if cell_count >= max_cells:
                    lines.append(f"[TRUNCATED] max_cells={max_cells}")
                    break
            if lines:
                out.append(("\n".join(lines), {**base_meta, "part": f"sheet_{sheet.title}"}))
    except Exception as e:
        return _skip("xlsx_iter_error", path, error=repr(e))
    finally:
        try:
            wb.close()
        except Exception:
            pass

    return out if out else [("", {**base_meta, "part": "empty"})]

def _xmind_walk_topic(topic: Dict[str, Any], acc: List[str], depth: int = 0):
    title = topic.get("title") or topic.get("text") or ""
    title = str(title).strip()
    if title:
        acc.append(("  " * depth) + "- " + title)

    children = None
    if isinstance(topic.get("children"), dict):
        children = topic["children"].get("attached") or topic["children"].get("topics")
    elif isinstance(topic.get("children"), list):
        children = topic["children"]

    if children:
        for ch in children:
            if isinstance(ch, dict):
                _xmind_walk_topic(ch, acc, depth + 1)

def read_xmind(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if not path.exists() or not path.is_file():
        return _skip("missing_file", path)

    if not zipfile.is_zipfile(str(path)):
        return _skip("xmind_not_a_zip", path)

    lines: List[str] = []
    base_meta = {"source": str(path)}

    try:
        with zipfile.ZipFile(path, "r") as z:
            names = set(z.namelist())

            if "content.json" in names:
                data = json.loads(z.read("content.json").decode("utf-8", errors="ignore"))
                if isinstance(data, list):
                    for si, sheet in enumerate(data, start=1):
                        root = sheet.get("rootTopic") or sheet.get("topic") or {}
                        sheet_title = sheet.get("title") or f"sheet_{si}"
                        lines.append(f"# {sheet_title}")
                        _xmind_walk_topic(root, lines, 0)
                elif isinstance(data, dict):
                    root = data.get("rootTopic") or data.get("topic") or data
                    _xmind_walk_topic(root, lines, 0)

            elif "content.xml" in names:
                xml_bytes = z.read("content.xml")
                root = ET.fromstring(xml_bytes)

                def walk_xml_topic(node: ET.Element, depth: int = 0):
                    title = node.attrib.get("title", "").strip()
                    if title:
                        lines.append(("  " * depth) + "- " + title)
                    for child in node:
                        if child.tag.endswith("topic"):
                            walk_xml_topic(child, depth + 1)
                        else:
                            for t in child.findall(".//"):
                                if t.tag.endswith("topic"):
                                    walk_xml_topic(t, depth + 1)

                for t in root.findall(".//"):
                    if t.tag.endswith("topic") and t.attrib.get("title"):
                        walk_xml_topic(t, 0)
                        break
            else:
                lines.append("[Unsupported XMind] content.json/content.xml not found")

    except Exception as e:
        return _skip("xmind_read_error", path, error=repr(e))

    text = "\n".join(lines).strip()
    return [(text, {**base_meta, "part": "mindmap"})] if text else [("", {**base_meta, "part": "empty"})]

def extract_text(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    ext = path.suffix.lower()
    if ext in {".md", ".txt"}:
        return read_txt_md(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pptx":
        return read_pptx(path)
    if ext == ".xlsx":
        return read_xlsx(path)
    if ext == ".xmind":
        return read_xmind(path)
    return [("", {"part": "unsupported", "source": str(path)})]

# ----------------- Qdrant REST helpers -----------------
def qdrant_request(method: str, url: str, body: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Qdrant HTTP {e.code}: {detail}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Qdrant unreachable at {QDRANT_URL}: {e}")

def ensure_collection(dim: int):
    r = qdrant_request("GET", f"{QDRANT_URL}/collections")
    names = {c["name"] for c in r.get("result", {}).get("collections", [])}
    if COLLECTION in names:
        return

    body = {"vectors": {"size": dim, "distance": "Cosine"}}
    qdrant_request("PUT", f"{QDRANT_URL}/collections/{COLLECTION}", body)

def upsert_points(points: List[Dict[str, Any]]):
    body = {"points": points}
    qdrant_request("PUT", f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true", body)

# ----------------- Main -----------------
def main(kb_dir: str = "../kb"):
    kb_path = Path(kb_dir).resolve()
    if not kb_path.exists():
        raise SystemExit(f"kb_dir introuvable: {kb_path}")

    print(f"📁 Ingest from: {kb_path}")
    print(f"🧠 Embeddings: model={EMBED_MODEL} | batch_size={BATCH_SIZE} | dedupe={DEDUPE}")

    files: List[Path] = []
    for p in kb_path.rglob("*"):
        if p.is_dir():
            continue
        if should_skip(p):
            continue
        if p.suffix.lower() not in INCLUDE_EXT:
            continue
        files.append(p)

    if not files:
        print("⚠️ Aucun fichier supporté trouvé (md/txt/docx/pptx/xlsx/xmind).")
        return

    # detect embedding dim
    dim = get_embedding_dim()
    ensure_collection(dim)

    total_chunks = 0
    total_skipped = 0
    batch_points: List[Dict[str, Any]] = []

    for f in files:
        rel = str(f.relative_to(kb_path))
        extracted = extract_text(f)

        # log skips (file parsing)
        for _, meta in extracted:
            if meta.get("part") == "skip":
                total_skipped += 1
                print(f"⚠️ SKIP {rel} -> {meta.get('reason')}")

        for text, meta in extracted:
            if meta.get("part") == "skip":
                continue
            if not text or not text.strip():
                continue

            chunks = chunk_text(text)
            if not chunks:
                continue

            for i in range(0, len(chunks), BATCH_SIZE):
                sub = chunks[i:i + BATCH_SIZE]
                vecs = embed_texts(sub, normalize=True)

                for chunk, vec in zip(sub, vecs):
                    if vec is None:
                        total_skipped += 1
                        continue

                    part = meta.get("part", "full")
                    pid = make_point_id(rel, part, chunk) if DEDUPE else str(uuid.uuid4())

                    batch_points.append({
                        "id": pid,
                        "vector": vec,
                        "payload": {
                            "source": rel,
                            "filetype": f.suffix.lower().lstrip("."),
                            "part": part,
                            "text": chunk
                        }
                    })
                    total_chunks += 1

                if len(batch_points) >= 256:
                    upsert_points(batch_points)
                    batch_points = []

    if batch_points:
        upsert_points(batch_points)

    print(f"✅ Ingested {total_chunks} chunks into '{COLLECTION}' (embed={EMBED_MODEL})")
    if total_skipped:
        print(f"ℹ️ Skipped {total_skipped} items due to read/format/embed errors.")

if __name__ == "__main__":
    main()
