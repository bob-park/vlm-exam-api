from __future__ import annotations

from pathlib import Path
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.db import AsyncSessionLocal
from app.models import Video, CatalogImage, FaceEmbedding
from app.utils.video import get_video_metadata, extract_catalog_images
from app.utils.clip_utils import get_clip_service
from app.utils.face_utils import get_face_service

logger = logging.getLogger(__name__)
settings = get_settings()
_CATALOG_PREFIX = "catalog_"


def _catalog_timestamp_seconds(path: Path) -> float | None:
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


async def process_video_task(video_id: int) -> None:
    async with AsyncSessionLocal() as session:
        await process_video(session, video_id)


def _safe_path(path: str) -> Path:
    return Path(path)


async def process_video(session: AsyncSession, video_id: int) -> None:
    clip_service = get_clip_service()
    face_service = get_face_service()

    video = await session.get(Video, video_id)
    if not video:
        logger.error("Video not found: %s", video_id)
        return

    try:
        video_path = _safe_path(video.path)
        duration, width, height = get_video_metadata(video_path)
        video.duration = duration
        video.width = width
        video.height = height

        catalog_dir = Path(settings.catalogs_dir) / f"video_{video.id}"
        catalogs = extract_catalog_images(video_path, catalog_dir, duration, width, height)

        for catalog_path, out_w, out_h in catalogs:
            catalog = CatalogImage(
                video_id=video.id,
                path=str(catalog_path),
                width=out_w,
                height=out_h,
            )
            session.add(catalog)
            await session.flush()

            ts = _catalog_timestamp_seconds(catalog_path)
            if ts is not None:
                logger.debug(
                    "Catalog text extraction at t=%.2fs (video_id=%s, catalog_id=%s, path=%s)",
                    ts,
                    video.id,
                    catalog.id,
                    catalog_path,
                )
            else:
                logger.debug(
                    "Catalog text extraction at t=unknown (video_id=%s, catalog_id=%s, path=%s)",
                    video.id,
                    catalog.id,
                    catalog_path,
                )

            caption = clip_service.caption_ko(catalog_path)
            image_embedding = clip_service.image_embedding(catalog_path)

            catalog.caption_ko = caption
            catalog.clip_image_embedding = image_embedding

            faces = face_service.extract_faces(catalog_path)
            for bbox, embedding in faces:
                session.add(
                    FaceEmbedding(
                        catalog_image_id=catalog.id,
                        bbox=bbox,
                        embedding=embedding,
                    )
                )

        video.status = "ready"
        await session.commit()

    except Exception as exc:
        logger.exception("Processing failed for video %s", video_id)
        video.status = "error"
        video.error_message = str(exc)
        await session.commit()
