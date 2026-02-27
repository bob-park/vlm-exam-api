from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional


class VideoOut(BaseModel):
    id: int
    filename: str
    content_type: Optional[str]
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    status: str
    error_message: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class CatalogImageOut(BaseModel):
    id: int
    video_id: int
    path: str
    width: int
    height: int
    caption_ko: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class SearchResult(BaseModel):
    catalog_image_id: int
    video_id: int
    caption_ko: Optional[str]
    score: float
    seconds: Optional[float] = None


class VideoSearchResponse(BaseModel):
    videos: List[VideoOut]


class TextSearchRequest(BaseModel):
    text: str
    top_k: Optional[int] = None


class TextSearchResponse(BaseModel):
    results: List[SearchResult]


class FaceSearchResponse(BaseModel):
    results: List[SearchResult]


class FaceDetectResponse(BaseModel):
    bboxes: List[List[int]]
