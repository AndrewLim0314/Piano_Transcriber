from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Music Transcription API"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    database_url: str = "sqlite+aiosqlite:///./transcription.db"

    upload_dir: str = "./uploads"
    output_dir: str = "./outputs"
    max_file_size: int = 52428800  # 50MB

    redis_url: str = "redis://localhost:6379/0"

    allowed_audio_formats: list = [".mp3", ".wav", ".ogg", ".flac", ".m4a"]

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()