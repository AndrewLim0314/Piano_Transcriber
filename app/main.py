from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from app.config import get_settings
from app.db.database import init_db
from app.routers import health, transcription
from app.utils.file_handler import ensure_output_dir

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    ensure_output_dir()
    yield
    # Shutdown
    pass


app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/files", StaticFiles(directory=settings.output_dir), name="files")

# Include routers
app.include_router(health.router)
app.include_router(transcription.router)


@app.get("/")
async def root():
    return {
        "message": "Music Transcription API",
        "docs": "/docs",
        "health": "/health"
    }