# NAS Document Indexer

Lokaler RAG-Dokumenten-Indexer für eine Synology NAS. PDFs werden per Ollama eingebettet, in Qdrant gespeichert und über eine FastAPI-Web-UI durchsuchbar gemacht.

## Architektur

```
/mnt/nas (NFS, read-only)
       │
  [indexer] ──embed──► Ollama (nomic-embed-text)
       │
       ▼
   [Qdrant] ◄──── [search-api] ──LLM──► Ollama (llama3.2:3b)
                       │
                  http://localhost:8765
```

## Voraussetzungen

- Docker & Docker Compose v2
- Ollama läuft auf dem Host
- NFS-Mount der NAS unter `/mnt/nas` (read-only)

## 1. Ollama-Modelle laden

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

## 2. NFS-Mount einrichten

Eintrag in `/etc/fstab` (Beispiel für Synology NAS):

```
192.168.1.216:/volume1/documents  /mnt/nas  nfs  ro,noatime,soft,timeo=30  0  0
```

Mount aktivieren:

```bash
sudo mkdir -p /mnt/nas
sudo mount /mnt/nas
ls /mnt/nas   # Verzeichnisinhalt prüfen
```

> **Wichtig:** Der Mount muss read-only (`ro`) sein. Der Indexer schreibt niemals auf `/mnt/nas`.

## 3. Konfiguration

```bash
cp .env.example .env
```

Mindestens anpassen:

| Variable | Bedeutung |
|---|---|
| `SYNOLOGY_IP` | IP-Adresse der NAS (für Web-Links) |
| `SMB_SHARE_NAME` | Name der SMB-Freigabe |
| `OLLAMA_HOST` | `172.17.0.1` (Linux) oder `host.docker.internal` (Docker Desktop) |

## 4. Starten

```bash
docker compose up -d --build
```

## 5. Logs beobachten

```bash
# Indexer-Fortschritt
docker compose logs -f indexer

# Alle Services
docker compose logs -f
```

Beim ersten Start werden alle PDFs indexiert — das kann je nach Anzahl und OCR-Bedarf einige Zeit dauern.

## 6. Web-UI

Browser öffnen: **http://localhost:8765**

- **Suche** — semantische Ähnlichkeitssuche in allen Dokumenten
- **Frage stellen** — RAG Q&A, Antworten auf Deutsch via llama3.2:3b
- **Info** — Statistiken der Qdrant-Collection

## 7. API-Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| GET | `/search?q=...&limit=8&folder=...&dedupe=true` | Semantische Suche |
| GET | `/ask?q=...&limit=5&folder=...` | RAG Q&A |
| GET | `/stats` | Collection-Info |
| GET | `/` | Web-UI |

### Antwortformat `/search`

```json
{
  "results": [
    {
      "score": 0.87,
      "filename": "dokument.pdf",
      "rel_path": "Ordner/Unterordner/dokument.pdf",
      "folder": "Ordner",
      "page": 3,
      "text": "...",
      "source": "native",
      "smb_link": "\\\\NAS\\documents\\Ordner\\dokument.pdf",
      "web_link": "http://192.168.1.216:5000/webman/..."
    }
  ],
  "total": 1
}
```

## 8. Links in Browsern

- **Web-Link** — öffnet die Datei im Synology File Station (DSM 7)
- **Netzwerkpfad (SMB)** — `\\NAS\share\...` zum Kopieren und in Windows Explorer einfügen
  - Firefox unter Windows/macOS öffnet `smb://`-Links direkt
  - Chrome benötigt einen Registry-Handler; den Pfad aus der UI kopieren und manuell im Explorer öffnen

## 9. Docker Desktop (Windows/macOS)

Auf Docker Desktop ist `172.17.0.1` nicht verfügbar. In `.env` setzen:

```
OLLAMA_HOST=host.docker.internal
```

## 10. Umgebungsvariablen-Referenz

| Variable | Standard | Beschreibung |
|---|---|---|
| `SCAN_PATH` | `/mnt/nas` | Zu scannender Pfad |
| `SCAN_INTERVAL` | `3600` | Scan-Intervall in Sekunden |
| `QDRANT_HOST` | `qdrant` | Qdrant-Hostname |
| `QDRANT_PORT` | `6333` | Qdrant-Port |
| `QDRANT_COLLECTION` | `documents` | Collection-Name |
| `OLLAMA_HOST` | `172.17.0.1` | Ollama-Host |
| `OLLAMA_PORT` | `11434` | Ollama-Port |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding-Modell |
| `LLM_MODEL` | `llama3.2:3b` | LLM für Q&A |
| `OCR_LANG` | `deu+eng` | Tesseract-Sprachen |
| `CHUNK_SIZE` | `600` | Chunk-Größe (Zeichen) |
| `CHUNK_OVERLAP` | `80` | Overlap (Zeichen) |
| `INDEXER_THREADS` | `2` | Parallele Indexier-Threads |
| `SYNOLOGY_IP` | `192.168.1.216` | NAS-IP für Web-Links |
| `SMB_SHARE_NAME` | `documents` | SMB-Freigabename |

## 11. Fehlerbehebung

**Indexer: "Waiting for Qdrant/Ollama"**  
Qdrant muss gestartet und Ollama auf dem Host erreichbar sein. Prüfen:
```bash
curl http://172.17.0.1:11434/api/tags
curl http://localhost:6333/healthz
```

**OCR sehr langsam**  
`INDEXER_THREADS=1` setzen und `dpi=150` im Code reduzieren (in `extract_text_ocr`).

**NFS-Berechtigungsfehler**  
Den NFS-Export auf der NAS auf `ro` (read-only) und passende UID/GID-Optionen prüfen. Der Container läuft als root (UID 0) — ggf. `no_root_squash` auf der NAS aktivieren oder `anonuid`/`anongid` setzen.

**Qdrant-Collection existiert nicht beim API-Start**  
Die search-api ist tolerant: `/stats` und `/search` liefern einen 502-Fehler bis die Collection durch den Indexer angelegt wurde. Einfach warten bis der erste Scan-Zyklus abgeschlossen ist.
