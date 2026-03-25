"""
YouTube AI Media Detector – FastAPI Backend  v1.2.0
Receives base64-encoded video frames from the Chrome extension,
runs the AI analysis pipeline, and returns a JSON result.

Endpoints:
    POST /analyze          – Single frame → instant result
    POST /analyze-batch    – Up to 20 frames → aggregated result (fewer round-trips)
    GET  /status           – Model load state + device info
    GET  /health           – Health check

Run:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

from analyzer import analyze_frame, _get_mtcnn, _get_clip, _get_deepfake_pipeline, DEVICE

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
)
logger = logging.getLogger(__name__)

# Track which models have successfully loaded
_loaded_models: dict[str, bool] = {
    "clip": False,
    "mtcnn": False,
    "deepfake_vit": False,
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing local AI models for live analysis...")
    try:
        _get_mtcnn()
        _loaded_models["mtcnn"] = True
        logger.info("MTCNN loaded ✓")
    except Exception as e:
        logger.warning(f"MTCNN failed to load: {e}")

    try:
        _get_clip()
        _loaded_models["clip"] = True
        logger.info("CLIP loaded ✓")
    except Exception as e:
        logger.warning(f"CLIP failed to load: {e}")

    try:
        pipe = _get_deepfake_pipeline()
        _loaded_models["deepfake_vit"] = pipe != "fallback"
        logger.info("Deepfake ViT loaded ✓" if _loaded_models["deepfake_vit"] else "Deepfake ViT using fallback")
    except Exception as e:
        logger.warning(f"Deepfake ViT failed to load: {e}")

    logger.info("Startup complete. Ready to analyze frames.")
    yield

# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    lifespan=lifespan,
    title="YouTube AI Media Detector",
    description=(
        "Analyzes video frames captured by the Chrome extension and classifies "
        "content as: real video | AI-generated | cartoon/animation | "
        "video game | deepfake detected."
    ),
    version="1.1.0",
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
    reason: str = Field(default="", description="Human-readable reason for classification")
    detail: str = Field(default="", description="Detailed error message if type is error")
    frames_analyzed: int = Field(default=1, description="Number of frames that contributed to this result")


class BatchFrameRequest(BaseModel):
    images: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="List of base64-encoded images (JPEG/PNG) — up to 20 frames.",
    )
    metadata: dict = Field(
        default={},
        description="Optional metadata like 'title' and 'description' of the video.",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check")
def root():
    return {"status": "ok", "service": "YouTube AI Media Detector", "version": "1.2.0"}


@app.get("/health", summary="Health check (alias)")
def health():
    return {"status": "ok"}


@app.get("/status", summary="Model load status")
def status():
    """Returns which AI models are loaded and the active compute device."""
    return {
        "status": "ok",
        "version": "1.2.0",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models": _loaded_models,
    }


@app.post("/analyze", response_model=AnalysisResult, summary="Analyze a single video frame")
def analyze(req: FrameRequest):
    """
    Accepts a single base64-encoded video frame, runs the full AI pipeline, and
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
        msg = result.get("detail") or "Error analyzing frame"
        logger.error(f"Analysis error: {msg}")
        return AnalysisResult(type="error", confidence=0.0, reason="", detail=str(msg), frames_analyzed=0)

    return AnalysisResult(
        type=result["type"],
        confidence=result["confidence"],
        reason=result.get("reason", ""),
        detail="",
        frames_analyzed=1,
    )


@app.post("/analyze-batch", response_model=AnalysisResult, summary="Analyze multiple frames at once")
def analyze_batch_endpoint(req: BatchFrameRequest):
    """
    Accepts up to 20 base64-encoded video frames in a single request, runs
    the full AI pipeline on all of them, and returns one aggregated result.

    Use this when you want to send several frames in a single HTTP call to
    reduce connection overhead (e.g., short videos where all frames are ready
    simultaneously).
    """
    if not req.images:
        raise HTTPException(status_code=400, detail="'images' list must not be empty.")

    from analyzer import analyze_batch as _analyze_batch
    result = _analyze_batch(req.images, metadata=req.metadata)

    n_frames = len(req.images)
    if result.get("type") == "error":
        msg = result.get("detail") or "Error analyzing batch"
        logger.error(f"Batch analysis error: {msg}")
        return AnalysisResult(type="error", confidence=0.0, reason="", detail=str(msg), frames_analyzed=0)

    return AnalysisResult(
        type=result["type"],
        confidence=result["confidence"],
        reason=result.get("reason", ""),
        detail="",
        frames_analyzed=n_frames,
    )
