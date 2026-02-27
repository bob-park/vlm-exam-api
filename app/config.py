from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_name: str = "vlm-exam-api"
    debug: bool = True

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/vlm"

    storage_dir: str = "./data"
    videos_dir: str = "./data/videos"
    catalogs_dir: str = "./data/catalogs"

    clip_model_name: str = "ViT-B/32"
    clip_device: str = "cpu"
    clip_model_path: str | None = "./data/models/ViT-B-32.pt"
    clip_download_root: str | None = None
    clip_ca_bundle: str | None = None
    clip_allow_download: bool = False

    insightface_model: str = "buffalo_l"
    insightface_device: str = "cpu"

    max_catalog_height: int = 1080
    max_catalog_width: int = 1920

    log_level: str = "INFO"

    top_k_default: int = 10

    @property
    def storage_path(self) -> Path:
        return Path(self.storage_dir)


@lru_cache
def get_settings() -> Settings:
    return Settings()
