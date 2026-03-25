"""
YouTube AI Media Detector - Core Analysis Engine
Powered by Local Models (CLIP, MTCNN, ViT)
"""

import io
import base64
import logging
import numpy as np
import json
import importlib.metadata
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import cv2
from pydantic import BaseModel, Field

# ──────────────────────────────────────────────────────────────────────────────
# Runtime Environment Fix (Self-Healing)
# ──────────────────────────────────────────────────────────────────────────────
# On some systems (especially those using OneDrive sync), the virtual environment 
# metadata can get corrupted, leading to 'importlib.metadata.version("huggingface_hub")'
# returning None. This causes 'transformers' to crash. We monkeypatch it here
# to provide a safe fallback version string if needed.
# ──────────────────────────────────────────────────────────────────────────────
import importlib.metadata
import sys

def _patch_metadata(module):
    if not hasattr(module, "version"): return
    _original = module.version
    def _patched(dist):
        try:
            v = _original(dist)
            if v: return v
        except: pass
        # Fallbacks for core AI components
        low = dist.lower().replace("_", "-")
        if "huggingface" in low: return "0.22.2"
        if "transformers" in low: return "4.38.0"
        if "accelerate" in low: return "0.26.0"
        if "tokenizers" in low: return "0.15.0"
        return "0.1.0"
    module.version = _patched

# Patch standard library
_patch_metadata(importlib.metadata)

# Patch backport if it exists in the environment
try:
    import importlib_metadata
    _patch_metadata(importlib_metadata)
except ImportError:
    pass

# Patch specific transformers utility to bypass the crash point directly
try:
    import transformers.utils.versions as _tv
    if hasattr(_tv, "require_version_core"):
        _orig_rvc = _tv.require_version_core
        def _patched_rvc(requirement):
            try:
                return _orig_rvc(requirement)
            except ValueError as e:
                if "huggingface-hub" in str(requirement): return
                raise e
        _tv.require_version_core = _patched_rvc
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Device Setup
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# CLIP scene-classification labels
# ──────────────────────────────────────────────────────────────────────────────
SCENE_LABELS = [
    "a high-quality photograph or realistic cinematic camera footage of a real-world scene, realistic textures, natural lighting, authentic video recording",
    "an AI-generated synthetic video frame, Sora video, DALL-E video, artificial textures, digital artifacts, distorted details, synthetic imagery, uncanny valley",
    "a cartoon, anime, 2D or 3D animation, stylized illustration, non-photorealistic render, Disney or Pixar style, hand-drawn or computer-animated characters",
    "computer-generated imagery from a video game, CGI gameplay, 3D engine render, low-poly or high-res gaming graphics, HUD interface, game UI elements",
]

LABEL_TO_TYPE = {
    0: "real_video",
    1: "ai_generated",
    2: "cartoon_animation",
    3: "video_game",
}

# ──────────────────────────────────────────────────────────────────────────────
# Local Model singletons
# ──────────────────────────────────────────────────────────────────────────────
_mtcnn: MTCNN | None = None
_clip_model = None
_clip_preprocess = None
_deepfake_pipeline = None

def _get_mtcnn() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        logger.info("Loading MTCNN face detector …")
        _mtcnn = MTCNN(
            keep_all=True, device=DEVICE,
            min_face_size=40, thresholds=[0.6, 0.7, 0.7], post_process=False
        )
    return _mtcnn

def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        logger.info("Loading CLIP ViT-B/32 …")
        try:
            import clip
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        except ImportError:
            from transformers import CLIPModel, CLIPProcessor
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            _clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_preprocess

def _get_deepfake_pipeline():
    global _deepfake_pipeline
    if _deepfake_pipeline is None:
        logger.info("Loading local deepfake ViT model …")
        try:
            from transformers import pipeline as hf_pipeline
            _deepfake_pipeline = hf_pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            logger.warning(f"Could not load deepfake ViT model: {e}")
            _deepfake_pipeline = "fallback"
    return _deepfake_pipeline

# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────
def decode_image(b64_data: str) -> Image.Image:
    """Decode a base64-encoded image string to a PIL Image."""
    if "," in b64_data:
        b64_data = b64_data.split(",", 1)[1]
    raw = base64.b64decode(b64_data)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return img


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    """Convert PIL Image → OpenCV BGR ndarray."""
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ──────────────────────────────────────────────────────────────────────────────
# Local Inference Logic
# ──────────────────────────────────────────────────────────────────────────────
def classify_scene(img: Image.Image) -> dict:
    model, preprocess = _get_clip()
    try:
        import clip
        image_input = preprocess(img).unsqueeze(0).to(DEVICE)
        text_inputs = clip.tokenize(SCENE_LABELS).to(DEVICE)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarity[0].cpu().numpy()
    except Exception:
        # HuggingFace CLIP fallback
        inputs = preprocess(text=SCENE_LABELS, images=img, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
    best_idx = int(np.argmax(probs))
    return {"type": LABEL_TO_TYPE[best_idx], "confidence": round(float(probs[best_idx]), 4)}

def detect_faces(img: Image.Image) -> list:
    mtcnn = _get_mtcnn()
    boxes, probs = mtcnn.detect(img)
    if boxes is None:
        return []
    faces = []
    w, h = img.size
    for box, prob in zip(boxes, probs):
        if prob and prob > 0.92:
            x1, y1, x2, y2 = [max(0, int(v)) for v in box]
            faces.append(img.crop((x1, y1, min(x2, w), min(y2, h))))
    return faces

def detect_deepfake(faces: list[Image.Image]) -> dict:
    pipe = _get_deepfake_pipeline()
    if pipe == "fallback":
        return {"type": "real_video", "confidence": 0.5}
    scores = []
    for face in faces:
        res = pipe(face.convert("RGB"))
        score_map = {r["label"].lower(): r["score"] for r in res}
        scores.append(score_map.get("fake", 1.0 - score_map.get("real", 0.5)))
    avg = float(np.mean(scores))
    if avg > 0.70:
        return {"type": "deepfake_detected", "confidence": round(avg, 4)}
    return {"type": "real_video", "confidence": round(1.0 - avg, 4)}

def analyze_text(metadata: dict) -> dict:
    """Improved metadata analysis with weighted scoring and negative filters."""
    title = metadata.get("title", "").lower()
    desc = metadata.get("description", "").lower()
    text = f"{title} {desc}"

    # 1. Negative Keywords (Signals that it's likely real/human-made)
    # If these are found, we reduce the suspicion score
    negatives = ["not ai", "100% real", "behind the scenes", "vlog", "irl", "raw footage", "no filters"]
    has_negative = any(neg in text for neg in negatives)

    # 2. Weighted AI Keywords
    # Primary: Strong indicators of AI generation
    primary_keywords = {
        "ai generated": 1.0, "ai-generated": 1.0, "created by ai": 1.0,
        "sora ai": 1.0, "sora-ai": 1.0, "openai sora": 1.0,
        "klingai": 1.0, "kling ai": 1.0, "luma dream machine": 1.0,
        "runway gen-2": 1.0, "runway gen2": 1.0, "pika labs": 1.0,
        "midjourney": 0.8, "stable diffusion": 0.8, "synthesia": 0.9,
        "heygen": 0.9, "cloned voice": 0.9, "deepfake": 0.9,
        "ai-assisted": 0.7, "ai assisted": 0.7, "ai voices": 0.7
    }
    
    # Secondary: Generic AI terms (need multiple to trigger high score)
    secondary_keywords = ["ai tool", "generative ai", "gen-ai", "artificial intelligence", "neural network"]

    score = 0.0
    found_primary = []
    
    # Check primary
    for kw, weight in primary_keywords.items():
        if kw in text:
            score = max(score, weight)
            found_primary.append(kw)

    # Check secondary (additive)
    found_secondary = [kw for kw in secondary_keywords if kw in text]
    if not found_primary and found_secondary:
        score = min(0.6, len(found_secondary) * 0.25)

    # Apply negative filter
    if has_negative:
        score *= 0.5

    if score > 0.6:
        reason = f"AI disclosure found: {', '.join(found_primary + found_secondary)}"
        return {"score": round(score, 2), "reason": reason}
    
    return {"score": 0.0, "reason": ""}


def analyze_frame(b64_image: str, metadata: dict | None = None) -> dict:
    """Local live analysis (No API). Returns dict with type, confidence, and reason."""
    if metadata is None:
        metadata = {}
    try:
        img = decode_image(b64_image)

        # 1. Fast metadata keyword check
        text_res = analyze_text(metadata)
        if text_res["score"] > 0.9:
            return {"type": "ai_generated", "confidence": 1.0, "reason": text_res["reason"]}

        # 2. CLIP scene classification
        clip_res = classify_scene(img)
        clip_res.setdefault("reason", f"CLIP classified as {clip_res['type']}")

        # 3. Face detection + deepfake check (only if scene looks like real video or AI)
        if clip_res["type"] in ("real_video", "ai_generated"):
            faces = detect_faces(img)
            if faces:
                df_res = detect_deepfake(faces)
                if df_res["type"] == "deepfake_detected":
                    df_res["reason"] = f"Deepfake indicators in {len(faces)} face(s)"
                    return df_res

        return clip_res
    except Exception as e:
        logger.exception("Local analysis failed")
        return {"type": "error", "confidence": 0.0, "reason": "", "detail": str(e)}


def analyze_batch(frames: list[str], metadata: dict | None = None) -> dict:
    """
    Analyze a batch of base64-encoded frames and return an aggregated result.
    Uses confidence-weighted voting across all frames.
    Used by evaluate.py for dataset evaluation.
    """
    if metadata is None:
        metadata = {}
    if not frames:
        return {"type": "error", "confidence": 0.0, "reason": "No frames provided", "detail": ""}

    results = []
    for frame in frames:
        try:
            result = analyze_frame(frame, metadata=metadata)
            if result.get("type") != "error":
                results.append(result)
        except Exception as e:
            logger.warning(f"Frame analysis failed in batch: {e}")

    if not results:
        return {"type": "error", "confidence": 0.0, "reason": "All frames failed", "detail": ""}

    # Confidence-weighted aggregation
    scores: dict[str, float] = {}
    for r in results:
        t = r["type"]
        scores[t] = scores.get(t, 0.0) + r.get("confidence", 0.0)

    best_type = max(scores, key=lambda k: scores[k])
    matching = [r for r in results if r["type"] == best_type]
    avg_conf = sum(r.get("confidence", 0.0) for r in matching) / len(matching)

    return {
        "type": best_type,
        "confidence": round(avg_conf, 4),
        "reason": matching[0].get("reason", ""),
    }
