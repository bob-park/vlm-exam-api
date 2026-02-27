from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging

import asyncio
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, literal

from app.config import get_settings
from app.db import init_db, get_session
from app.logging_setup import setup_logging
from app.models import Video, CatalogImage, FaceEmbedding
from app.schemas import (
    VideoOut,
    VideoSearchResponse,
    TextSearchRequest,
    TextSearchResponse,
    FaceSearchResponse,
    FaceDetectResponse,
    SearchResult,
)
from app.services.processing import process_video_task
from app.utils.clip_utils import get_clip_service
from app.utils.face_utils import get_face_service

settings = get_settings()

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
_CATALOG_PREFIX = "catalog_"


def _catalog_seconds_from_path(path_str: str) -> float | None:
    path = Path(path_str)
    stem = path.stem
    if not stem.startswith(_CATALOG_PREFIX):
        return None
    idx_str = stem[len(_CATALOG_PREFIX) :]
    if not idx_str.isdigit():
        return None
    idx = int(idx_str)
    if idx <= 0:
        return None
    return float(idx - 1)


@app.on_event("startup")
async def on_startup() -> None:
    Path(settings.videos_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.catalogs_dir).mkdir(parents=True, exist_ok=True)
    await init_db()


@app.post("/videos/upload", response_model=VideoOut)
async def upload_video(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> VideoOut:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    video_path = Path(settings.videos_dir) / file.filename
    with video_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    video = Video(
        filename=file.filename,
        path=str(video_path),
        content_type=file.content_type,
        status="processing",
    )
    session.add(video)
    await session.commit()
    await session.refresh(video)

    asyncio.create_task(process_video_task(video.id))

    return VideoOut.model_validate(video)


@app.get("/videos", response_model=VideoSearchResponse)
async def list_videos(
    query: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session),
) -> VideoSearchResponse:
    stmt = select(Video)
    if query:
        stmt = stmt.where(Video.filename.ilike(f"%{query}%"))
    result = await session.execute(stmt.order_by(Video.created_at.desc()))
    videos = result.scalars().all()
    return VideoSearchResponse(videos=videos)


@app.get("/videos/{video_id}/stream")
async def stream_video(
    video_id: int,
    request: Request,
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    video = await session.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    path = Path(video.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video file missing")

    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    media_type = video.content_type or "video/mp4"

    if range_header:
        try:
            units, range_spec = range_header.split("=", 1)
        except ValueError:
            raise HTTPException(status_code=416, detail="Invalid Range header")
        if units.strip().lower() != "bytes":
            raise HTTPException(status_code=416, detail="Only bytes ranges are supported")

        start_str, end_str = range_spec.split("-", 1)
        if start_str == "":
            # suffix range: bytes=-N
            length = int(end_str)
            if length <= 0:
                raise HTTPException(status_code=416, detail="Invalid Range header")
            start = max(file_size - length, 0)
            end = file_size - 1
        else:
            start = int(start_str)
            end = int(end_str) if end_str else file_size - 1

        if start < 0 or end < start or start >= file_size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")
        end = min(end, file_size - 1)
        content_length = end - start + 1

        def iterfile():
            with path.open("rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        headers = {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(content_length),
        }
        return StreamingResponse(iterfile(), status_code=206, headers=headers, media_type=media_type)

    def iterfile():
        with path.open("rb") as f:
            while chunk := f.read(1024 * 1024):
                yield chunk

    return StreamingResponse(iterfile(), media_type=media_type, headers={"Accept-Ranges": "bytes"})


@app.get("/catalogs/{catalog_id}/thumbnail")
async def get_thumbnail(
    catalog_id: int,
    session: AsyncSession = Depends(get_session),
) -> FileResponse:
    catalog = await session.get(CatalogImage, catalog_id)
    if not catalog:
        raise HTTPException(status_code=404, detail="Catalog not found")
    path = Path(catalog.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail missing")
    media_type = "image/png"
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        media_type = "image/jpeg"
    return FileResponse(path, media_type=media_type)


@app.post("/search/text", response_model=TextSearchResponse)
async def search_by_text(
    req: TextSearchRequest,
    session: AsyncSession = Depends(get_session),
) -> TextSearchResponse:
    clip_service = get_clip_service()
    embedding = clip_service.text_embedding(req.text)
    top_k = req.top_k or settings.top_k_default

    distance = CatalogImage.clip_image_embedding.cosine_distance(embedding)
    stmt = (
        select(
            CatalogImage.id.label("catalog_image_id"),
            CatalogImage.video_id.label("video_id"),
            CatalogImage.caption_ko.label("caption_ko"),
            CatalogImage.path.label("path"),
            (literal(1.0) - distance).label("score"),
        )
        .where(CatalogImage.clip_image_embedding.is_not(None))
        .order_by(distance)
        .limit(top_k)
    )
    result = await session.execute(stmt)

    results = [
        SearchResult(
            catalog_image_id=row.catalog_image_id,
            video_id=row.video_id,
            caption_ko=row.caption_ko,
            score=float(row.score),
            seconds=_catalog_seconds_from_path(row.path),
        )
        for row in result.all()
    ]

    return TextSearchResponse(results=results)


@app.post("/search/face", response_model=FaceSearchResponse)
async def search_by_face(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session),
) -> FaceSearchResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    temp_path = Path(settings.storage_dir) / f"query_{file.filename}"
    with temp_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        face_service = get_face_service()
        faces = face_service.extract_faces(temp_path)
        if not faces:
            raise HTTPException(status_code=400, detail="No face detected")

        embedding = faces[0][1]
        top_k = settings.top_k_default

        distance = FaceEmbedding.embedding.cosine_distance(embedding)
        stmt = (
            select(
                FaceEmbedding.catalog_image_id.label("catalog_image_id"),
                CatalogImage.video_id.label("video_id"),
                CatalogImage.caption_ko.label("caption_ko"),
                CatalogImage.path.label("path"),
                (literal(1.0) - distance).label("score"),
            )
            .join(CatalogImage, CatalogImage.id == FaceEmbedding.catalog_image_id)
            .order_by(distance)
            .limit(top_k)
        )
        result = await session.execute(stmt)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    results = [
        SearchResult(
            catalog_image_id=row.catalog_image_id,
            video_id=row.video_id,
            caption_ko=row.caption_ko,
            score=float(row.score),
            seconds=_catalog_seconds_from_path(row.path),
        )
        for row in result.all()
    ]

    return FaceSearchResponse(results=results)


@app.post("/faces/detect", response_model=FaceDetectResponse)
async def detect_faces(
    file: UploadFile = File(...),
) -> FaceDetectResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    temp_path = Path(settings.storage_dir) / f"detect_{file.filename}"
    with temp_path.open("wb") as f:
        content = await file.read()
        f.write(content)

    try:
        face_service = get_face_service()
        bboxes = face_service.extract_face_bboxes(temp_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return FaceDetectResponse(bboxes=bboxes)
