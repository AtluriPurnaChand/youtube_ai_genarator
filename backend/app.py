"""
YouTube AI Media Detector – FastAPI Backend
Receives base64-encoded video frames from the Chrome extension,
runs the AI analysis pipeline, and returns a JSON result.

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from analyzer import analyze_frame

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="YouTube AI Media Detector",
    description=(
        "Analyzes video frames captured by the Chrome extension and classifies "
        "content as: real video | AI-generated | cartoon/animation | "
        "video game | deepfake detected."
    ),
    version="1.0.0",
)

# Allow requests from any Chrome extension origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ──────────────────────────────────────────────────────────────────────────────

class FrameRequest(BaseModel):
    image: str = Field(
        ...,
        description="Base64-encoded image (JPEG/PNG), optionally prefixed with a data-URI header.",
    )
    metadata: dict = Field(
        default={},
        description="Optional metadata like 'title' and 'description' of the video.",
    )


class AnalysisResult(BaseModel):
    type: str = Field(
        ...,
        description=(
            "Detected content type: real_video | ai_generated | "
            "cartoon_animation | video_game | deepfake_detected | error"
        ),
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "YouTube AI Media Detector"}


@app.get("/health", summary="Health check (alias)")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResult, summary="Analyze a video frame")
async def analyze(req: FrameRequest):
    """
    Accepts a base64-encoded video frame, runs the full AI pipeline, and
    returns the detected content type with a confidence score.

    Content types:
    - **real_video**        – Live / authentic footage
    - **ai_generated**      – AI-synthesized video
    - **cartoon_animation** – Cartoon or animated content
    - **video_game**        – Video game / CGI footage
    - **deepfake_detected** – Human face(s) detected with deepfake artifacts
    - **error**             – Processing failure
    """
    if not req.image:
        raise HTTPException(status_code=400, detail="'image' field is required.")

    result = analyze_frame(req.image, metadata=req.metadata)

    if result.get("type") == "error":
        logger.error(f"Analysis error: {result.get('detail')}")
        # Still return 200 so the extension can display "error" gracefully
        return AnalysisResult(type="error", confidence=0.0)

    return AnalysisResult(
        type=result["type"],
        confidence=result["confidence"],
    )
