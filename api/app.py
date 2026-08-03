import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import redis
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS", "https://zshureih.github.io,http://localhost:5173"
).split(",")
MAX_QUEUE_DEPTH = int(os.environ.get("MAX_QUEUE_DEPTH", 20))
SUBMIT_RATE_LIMIT = os.environ.get("SUBMIT_RATE_LIMIT", "5/hour")

QUEUE_KEY = "video_queue:pending"


def video_key(request_id: str) -> str:
    return f"video:{request_id}"


r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="MOVA Queue API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class VideoRequest(BaseModel):
    artist: str = Field(..., min_length=1, max_length=200)
    song_title: str = Field(..., min_length=1, max_length=200)
    prompt: str = Field(
        "in focus, high res, artistic shots from a music video", max_length=500
    )
    audio_strength: float = Field(1.0, ge=0.0, le=1.0)
    alpha: float = Field(0.5, ge=0.0, le=1.0)
    noise_level: float = Field(0.0, ge=0.0, le=1.0)
    ddim_sampling_steps: int = Field(50, ge=1, le=150)
    ddim_eta: float = Field(0.0, ge=0.0, le=1.0)
    interpolation_steps: int = Field(1, ge=1, le=10)
    height: int = Field(512, ge=64, le=1024)
    width: int = Field(512, ge=64, le=1024)
    seed: int = Field(316199, ge=0)


class VideoStatus(BaseModel):
    id: str
    status: str  # queued | processing | completed | failed
    submitted_at: str
    data: VideoRequest
    result: Optional[str] = None
    error: Optional[str] = None
    note: Optional[str] = None


@app.get("/")
async def read_root():
    return {"message": "MOVA queue API"}


@app.get("/queue")
async def queue_depth():
    return {"pending": r.llen(QUEUE_KEY)}


@app.post("/video", status_code=202)
@limiter.limit(SUBMIT_RATE_LIMIT)
async def request_video(request: Request, body: VideoRequest):
    pending = r.llen(QUEUE_KEY)
    if pending >= MAX_QUEUE_DEPTH:
        raise HTTPException(status_code=503, detail="Queue is full, try again later")

    request_id = str(uuid.uuid4())
    record = VideoStatus(
        id=request_id,
        status="queued",
        submitted_at=datetime.now(timezone.utc).isoformat(),
        data=body,
    )
    r.set(video_key(request_id), record.model_dump_json())
    r.rpush(QUEUE_KEY, request_id)
    return {"id": request_id, "status": "queued"}


@app.get("/video/{request_id}", response_model=VideoStatus)
async def get_video(request_id: str):
    raw = r.get(video_key(request_id))
    if raw is None:
        raise HTTPException(status_code=404, detail="Not found")
    return json.loads(raw)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
