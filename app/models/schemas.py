from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class TranscriptionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptionRequest(BaseModel):
    instrument: Optional[str] = Field(default="piano", description="Primary instrument")
    key_signature: Optional[str] = Field(default=None, description="Key signature hint")
    time_signature: Optional[str] = Field(default=None, description="Time signature hint")


class Note(BaseModel):
    pitch: str
    start_time: float
    duration: float
    velocity: int


class TranscriptionResponse(BaseModel):
    id: str
    status: TranscriptionStatus
    filename: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    midi_url: Optional[str] = None
    musicxml_url: Optional[str] = None
    pdf_url: Optional[str] = None
    notes: Optional[List[Note]] = None
    error: Optional[str] = None


class TranscriptionListResponse(BaseModel):
    total: int
    items: List[TranscriptionResponse]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str