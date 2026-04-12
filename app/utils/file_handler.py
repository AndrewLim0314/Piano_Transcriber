import os
import uuid
import aiofiles
from pathlib import Path
from fastapi import UploadFile, HTTPException
from app.config import get_settings

settings = get_settings()


def get_file_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_audio_file(filename: str) -> bool:
    ext = get_file_extension(filename)
    return ext in settings.allowed_audio_formats


async def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    """Save uploaded file and return (file_id, file_path)"""

    if not validate_audio_file(upload_file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed: {settings.allowed_audio_formats}"
        )

    # Create upload directory if not exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Generate unique filename
    file_id = str(uuid.uuid4())
    ext = get_file_extension(upload_file.filename)
    file_path = os.path.join(settings.upload_dir, f"{file_id}{ext}")

    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await upload_file.read()

        if len(content) > settings.max_file_size:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max size: {settings.max_file_size / 1024 / 1024}MB"
            )

        await f.write(content)

    return file_id, file_path


def ensure_output_dir():
    os.makedirs(settings.output_dir, exist_ok=True)