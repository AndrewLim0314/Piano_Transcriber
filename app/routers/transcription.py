from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime, timezone
import json

from app.models.schemas import (
    TranscriptionResponse,
    TranscriptionListResponse,
    TranscriptionStatus
)
from app.db.database import get_db, Transcription, TranscriptionStatusDB
from app.services.transcription_service import TranscriptionService
from app.utils.file_handler import save_upload_file

router = APIRouter(prefix="/transcription", tags=["transcription"])
transcription_service = TranscriptionService()


async def process_transcription_task(
        transcription_id: str,
        file_path: str,
        instrument: str,  # Keep this for the function signature
        db_session: AsyncSession
):
    """Background task for processing transcription"""
    try:
        # Update status to processing
        result = await db_session.execute(
            select(Transcription).where(Transcription.id == transcription_id)
        )
        transcription = result.scalar_one()
        transcription.status = TranscriptionStatusDB.PROCESSING
        await db_session.commit()

        # Process transcription - REMOVE instrument parameter here
        result = transcription_service.process_transcription(
            transcription_id,
            file_path
            # Removed: instrument
        )

        # Update database
        transcription.status = TranscriptionStatusDB.COMPLETED
        transcription.completed_at = datetime.now(timezone.utc)
        transcription.midi_path = result['midi_path']
        transcription.musicxml_path = result['musicxml_path']
        transcription.pdf_path = result['pdf_path']
        transcription.notes_json = json.dumps(result['notes'])

        await db_session.commit()

    except Exception as e:
        # Update status to failed
        result = await db_session.execute(
            select(Transcription).where(Transcription.id == transcription_id)
        )
        transcription = result.scalar_one()
        transcription.status = TranscriptionStatusDB.FAILED
        transcription.error = str(e)
        await db_session.commit()


@router.post("/upload", response_model=TranscriptionResponse)
async def upload_audio(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        instrument: str = "piano",
        db: AsyncSession = Depends(get_db)
):
    """Upload audio file for transcription"""

    # Validate filename
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    # Save file
    file_id, file_path = await save_upload_file(file)

    # Create database record
    transcription = Transcription(
        id=file_id,
        filename=file.filename,  # Now safe - validated above
        file_path=file_path,
        instrument=instrument,
        status=TranscriptionStatusDB.PENDING
    )

    db.add(transcription)
    await db.commit()
    await db.refresh(transcription)

    # Add background task
    background_tasks.add_task(
        process_transcription_task,
        file_id,
        file_path,
        instrument,
        db
    )

    return TranscriptionResponse(
        id=transcription.id,
        status=TranscriptionStatus.PENDING,
        filename=transcription.filename,
        created_at=transcription.created_at
    )


@router.get("/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription(
        transcription_id: str,
        db: AsyncSession = Depends(get_db)
):
    """Get transcription by ID"""
    result = await db.execute(
        select(Transcription).where(Transcription.id == transcription_id)
    )
    transcription = result.scalar_one_or_none()

    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")

    notes = None
    if transcription.notes_json:
        notes = json.loads(transcription.notes_json)

    return TranscriptionResponse(
        id=transcription.id,
        status=TranscriptionStatus(transcription.status.value),
        filename=transcription.filename,
        created_at=transcription.created_at,
        completed_at=transcription.completed_at,
        midi_url=f"/files/{transcription.id}.mid" if transcription.midi_path is not None else None,
        musicxml_url=f"/files/{transcription.id}.musicxml" if transcription.musicxml_path is not None else None,
        pdf_url=f"/files/{transcription.id}.pdf" if transcription.pdf_path is not None else None,
        notes=notes,
        error=transcription.error
    )


@router.get("/", response_model=TranscriptionListResponse)
async def list_transcriptions(
        skip: int = 0,
        limit: int = 10,
        db: AsyncSession = Depends(get_db)
):
    """List all transcriptions"""
    result = await db.execute(
        select(Transcription).offset(skip).limit(limit)
    )
    transcriptions = result.scalars().all()

    items = []
    for t in transcriptions:
        items.append(TranscriptionResponse(
            id=t.id,
            status=TranscriptionStatus(t.status.value),
            filename=t.filename,
            created_at=t.created_at,
            completed_at=t.completed_at,
            midi_url=f"/files/{t.id}.mid" if t.midi_path else None,
            musicxml_url=f"/files/{t.id}.musicxml" if t.musicxml_path else None,
            pdf_url=f"/files/{t.id}.pdf" if t.pdf_path else None
        ))

    return TranscriptionListResponse(
        total=len(items),
        items=items
    )


@router.delete("/{transcription_id}")
async def delete_transcription(
        transcription_id: str,
        db: AsyncSession = Depends(get_db)
):
    """Delete transcription"""
    result = await db.execute(
        select(Transcription).where(Transcription.id == transcription_id)
    )
    transcription = result.scalar_one_or_none()

    if not transcription:
        raise HTTPException(status_code=404, detail="Transcription not found")

    await db.delete(transcription)
    await db.commit()

    return {"message": "Transcription deleted successfully"}