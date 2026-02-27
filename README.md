# vlm-exam-api

FastAPI service for video ingestion, catalog image extraction, CLIP-based captioning/embeddings, and face embeddings with pgvector.

## Run (Docker)

1) Copy `.env.example` to `.env` and adjust if needed.
2) Build and run:

```bash
docker compose up --build
```

API: `http://localhost:8000`

## Run (Local)

Requirements:
- Python 3.10+
- ffmpeg
- PostgreSQL with pgvector extension

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

- `POST /videos/upload` (multipart file)
- `GET /videos?query=...`
- `GET /videos/{video_id}/stream`
- `GET /catalogs/{catalog_id}/thumbnail`
- `POST /search/text` (JSON: `{ "text": "...", "top_k": 10 }`)
- `POST /search/face` (multipart image)
