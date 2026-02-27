from __future__ import annotations

from pathlib import Path
import os
from typing import List
from functools import lru_cache
import logging

import torch
from PIL import Image
import clip

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ClipService:
    def __init__(self) -> None:
        self.device = settings.clip_device
        model_id = settings.clip_model_name
        if settings.clip_model_path:
            model_path = Path(settings.clip_model_path)
            if model_path.exists():
                model_id = str(model_path)
            elif settings.clip_allow_download:
                logger.warning(
                    "CLIP model file not found at %s. Falling back to download.",
                    model_path,
                )
            else:
                raise FileNotFoundError(
                    f"CLIP model path does not exist: {model_path}. "
                    "Set CLIP_MODEL_PATH to a valid local file or set "
                    "CLIP_ALLOW_DOWNLOAD=true to download."
                )

        if settings.clip_ca_bundle:
            os.environ.setdefault("SSL_CERT_FILE", settings.clip_ca_bundle)

        self.model, self.preprocess = clip.load(
            model_id,
            device=self.device,
            download_root=settings.clip_download_root,
        )

    def image_embedding(self, image_path: Path) -> List[float]:
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy()[0].tolist()

    def text_embedding(self, text: str) -> List[float]:
        text_tokens = clip.tokenize([text]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy()[0].tolist()

    def caption_ko(self, image_path: Path) -> str:
        labels = [
            "사람",
            "여러 명의 사람",
            "남성",
            "여성",
            "아이",
            "동물",
            "고양이",
            "개",
            "자동차",
            "버스",
            "기차",
            "비행기",
            "자연 풍경",
            "산",
            "바다",
            "도시",
            "거리",
            "건물",
            "실내",
            "음식",
            "축구",
            "농구",
            "야구",
            "사람들이 모여 있는 장면",
            "한 사람이 있는 장면",
            "야외 장면",
            "밤",
            "낮",
            "노을",
            "눈",
        ]
        prompts = [f"이 이미지는 {label}가 있는 장면이다." for label in labels]

        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)
            topk = similarities.topk(3).indices.cpu().numpy().tolist()

        top_labels = [labels[i] for i in topk]
        caption = f"이 이미지는 {', '.join(top_labels)}가 포함된 장면입니다."
        logger.info("Generated caption: %s", caption)
        return caption


@lru_cache
def get_clip_service() -> ClipService:
    return ClipService()
