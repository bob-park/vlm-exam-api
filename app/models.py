from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, Float
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    filename = Column(String(512), nullable=False)
    path = Column(String(1024), nullable=False)
    content_type = Column(String(128), nullable=True)

    duration = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    status = Column(String(32), nullable=False, default="processing")
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    catalogs = relationship("CatalogImage", back_populates="video", cascade="all, delete-orphan")


class CatalogImage(Base):
    __tablename__ = "catalog_images"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)

    path = Column(String(1024), nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)

    caption_ko = Column(Text, nullable=True)
    clip_image_embedding = Column(Vector(512), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    video = relationship("Video", back_populates="catalogs")
    faces = relationship("FaceEmbedding", back_populates="catalog_image", cascade="all, delete-orphan")


class FaceEmbedding(Base):
    __tablename__ = "face_embeddings"

    id = Column(Integer, primary_key=True)
    catalog_image_id = Column(Integer, ForeignKey("catalog_images.id", ondelete="CASCADE"), nullable=False)

    bbox = Column(String(128), nullable=False)
    embedding = Column(Vector(512), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    catalog_image = relationship("CatalogImage", back_populates="faces")
