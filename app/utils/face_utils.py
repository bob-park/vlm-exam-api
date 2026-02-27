from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from functools import lru_cache
import logging

import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class FaceService:
    def __init__(self) -> None:
        if settings.insightface_device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name=settings.insightface_model, providers=providers)
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def extract_faces(self, image_path: Path) -> List[Tuple[str, List[float]]]:
        image = Image.open(image_path).convert("RGB")
        img = np.asarray(image)

        faces = self.app.get(img)
        results: List[Tuple[str, List[float]]] = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            bbox_str = ",".join(str(v) for v in bbox)
            embedding = face.embedding.astype(float).tolist()
            results.append((bbox_str, embedding))

        logger.info("Extracted %d faces", len(results))
        return results

    def extract_face_bboxes(self, image_path: Path) -> List[List[int]]:
        image = Image.open(image_path).convert("RGB")
        img = np.asarray(image)

        faces = self.app.get(img)
        results: List[List[int]] = []
        for face in faces:
            bbox = face.bbox.astype(int).tolist()
            results.append([int(v) for v in bbox])

        logger.info("Extracted %d face bboxes", len(results))
        return results


@lru_cache
def get_face_service() -> FaceService:
    return FaceService()
