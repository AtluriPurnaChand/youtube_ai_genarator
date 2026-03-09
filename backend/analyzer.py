"""
YouTube AI Media Detector - Core Analysis Engine
Handles face detection, deepfake detection, and scene classification.
"""

import io
import base64
import logging
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import cv2

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Device Setup
# ──────────────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────────────────────────
# CLIP scene-classification labels
# Optimized for distinguishing AI textures vs real-world camera noise
# ──────────────────────────────────────────────────────────────────────────────
SCENE_LABELS = [
    "a high-quality photograph or realistic cinematic camera footage of a real-world scene, realistic textures, natural lighting",
    "an AI-generated synthetic video frame, Sora video, artificial textures, digital artifacts, distorted details, synthetic imagery",
    "a cartoon, anime, 2D or 3D animation, stylized illustration, non-photorealistic",
    "computer-generated imagery from a video game, CGI gameplay, 3D engine render, low-poly or high-res gaming graphics",
]

LABEL_TO_TYPE = {
    0: "real_video",
    1: "ai_generated",
    2: "cartoon_animation",
    3: "video_game",
}

# ──────────────────────────────────────────────────────────────────────────────
# Model singletons  (lazy-loaded once on first use)
# ──────────────────────────────────────────────────────────────────────────────
_mtcnn: MTCNN | None = None
_clip_model = None
_clip_preprocess = None
_deepfake_pipeline = None   # HuggingFace image-classification pipeline


def _get_mtcnn() -> MTCNN:
    global _mtcnn
    if _mtcnn is None:
        logger.info("Loading MTCNN face detector …")
        _mtcnn = MTCNN(
            keep_all=True,
            device=DEVICE,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            post_process=False,
        )
    return _mtcnn


def _get_clip():
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        logger.info("Loading CLIP ViT-B/32 …")
        try:
            import clip  # openai-clip
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        except ImportError:
            from transformers import CLIPModel, CLIPProcessor
            _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
            _clip_preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            _clip_model._hf = True
    return _clip_model, _clip_preprocess


def _get_deepfake_pipeline():
    """
    Loads a ViT model that was ACTUALLY trained for deepfake detection.
    Model: dima806/deepfake_vs_real_image_detection
    - Fine-tuned ViT-base-patch16-224 on 190k real vs deepfake face images
    - Labels: "Fake" / "Real"
    - Source: https://huggingface.co/dima806/deepfake_vs_real_image_detection
    Falls back to a CLIP-based heuristic if the model cannot be loaded.
    """
    global _deepfake_pipeline
    if _deepfake_pipeline is None:
        logger.info("Loading deepfake ViT model (dima806/deepfake_vs_real_image_detection) …")
        try:
            from transformers import pipeline as hf_pipeline
            _deepfake_pipeline = hf_pipeline(
                "image-classification",
                model="dima806/deepfake_vs_real_image_detection",
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("Deepfake ViT model loaded ✓")
        except Exception as e:
            logger.warning(f"Could not load deepfake ViT model: {e}. Using CLIP fallback.")
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
# Scene classification (CLIP)
# ──────────────────────────────────────────────────────────────────────────────
def classify_scene(img: Image.Image) -> dict:
    """
    Use CLIP to classify the scene into one of the four content types.
    Returns {"type": str, "confidence": float}.
    """
    model, preprocess = _get_clip()

    try:
        import clip as openai_clip
        image_input = preprocess(img).unsqueeze(0).to(DEVICE)
        text_inputs = openai_clip.tokenize(SCENE_LABELS).to(DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features  = model.encode_text(text_inputs)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features  /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        probs = similarity[0].cpu().numpy()

    except (ImportError, AttributeError):
        from transformers import CLIPModel, CLIPProcessor
        inputs = preprocess(
            text=SCENE_LABELS,
            images=img,
            return_tensors="pt",
            padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            probs = logits.softmax(dim=1)[0].cpu().numpy()

    best_idx   = int(np.argmax(probs))
    confidence = float(probs[best_idx])

    return {
        "type":       LABEL_TO_TYPE[best_idx],
        "confidence": round(confidence, 4),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Face detection (MTCNN)
# ──────────────────────────────────────────────────────────────────────────────
def detect_faces(img: Image.Image):
    """
    Run MTCNN on the image.
    Returns a list of cropped face PIL Images (may be empty).
    """
    mtcnn  = _get_mtcnn()
    boxes, probs = mtcnn.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    faces = []
    w, h  = img.size
    for box, prob in zip(boxes, probs):
        if prob is None or prob < 0.92:   # strict: only high-confidence faces
            continue
        x1, y1, x2, y2 = [max(0, int(v)) for v in box]
        x2 = min(x2, w)
        y2 = min(y2, h)
        if x2 > x1 and y2 > y1:
            faces.append(img.crop((x1, y1, x2, y2)))

    return faces


# ──────────────────────────────────────────────────────────────────────────────
# Deepfake detection  (real trained ViT model)
# ──────────────────────────────────────────────────────────────────────────────
def detect_deepfake(faces: list[Image.Image]) -> dict:
    """
    Run deepfake inference on detected face crops using a HuggingFace ViT
    model that was actually trained on deepfake vs real face datasets.

    Model: dima806/deepfake_vs_real_image_detection
    - Architecture : ViT-base-patch16-224 (fine-tuned)
    - Training data: ~190 000 images (Midjourney, StyleGAN, DALL-E, real faces)
    - Labels       : "Fake" (deepfake) / "Real"
    - Accuracy     : ~99 % on held-out test set

    Returns {"type": "deepfake_detected" | "real_video", "confidence": float}.
    """
    pipe = _get_deepfake_pipeline()

    fake_scores = []

    for face in faces:
        face_rgb = face.convert("RGB")

        # ── Primary path: ViT deepfake model ──────────────────────────────────
        if pipe != "fallback":
            try:
                results = pipe(face_rgb)
                # results is a list like [{"label": "Fake", "score": 0.97}, ...]
                score_map = {r["label"].lower(): r["score"] for r in results}
                fake_score = score_map.get("fake", 1.0 - score_map.get("real", 0.5))
                fake_scores.append(float(fake_score))
                logger.debug(f"ViT deepfake score: {fake_score:.4f}")
                continue
            except Exception as e:
                logger.warning(f"ViT deepfake inference failed for face crop: {e}")

        # ── Fallback path: CLIP-based deepfake prompts ─────────────────────
        # Used only if the ViT model failed to load or inference errored.
        try:
            model, preprocess = _get_clip()
            deepfake_labels = [
                "a real authentic human face photograph",
                "a deepfake AI generated synthetic human face",
            ]
            try:
                import clip as openai_clip
                img_input   = preprocess(face_rgb).unsqueeze(0).to(DEVICE)
                txt_input   = openai_clip.tokenize(deepfake_labels).to(DEVICE)
                with torch.no_grad():
                    img_feat = model.encode_image(img_input)
                    txt_feat = model.encode_text(txt_input)
                    img_feat /= img_feat.norm(dim=-1, keepdim=True)
                    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
                    sim = (100.0 * img_feat @ txt_feat.T).softmax(dim=-1)
                fake_score = float(sim[0][1].cpu())
            except (ImportError, AttributeError):
                from transformers import CLIPModel, CLIPProcessor
                inputs = preprocess(
                    text=deepfake_labels, images=face_rgb,
                    return_tensors="pt", padding=True,
                ).to(DEVICE)
                with torch.no_grad():
                    outputs  = model(**inputs)
                    probs    = outputs.logits_per_image.softmax(dim=1)[0].cpu().numpy()
                fake_score = float(probs[1])
            fake_scores.append(fake_score)
        except Exception as clip_err:
            logger.warning(f"CLIP fallback also failed: {clip_err}")
            # Last resort: cannot determine — assume real
            fake_scores.append(0.2)

    if not fake_scores:
        return {"type": "real_video", "confidence": 0.8}

    avg_score = float(np.mean(fake_scores))
    logger.info(f"Avg deepfake score across {len(fake_scores)} face(s): {avg_score:.4f}")

    # Threshold: flag deepfake only when the trained model is confident
    DEEPFAKE_THRESHOLD = 0.65
    if avg_score > DEEPFAKE_THRESHOLD:
        return {"type": "deepfake_detected", "confidence": round(avg_score, 4)}
    else:
        return {"type": "real_video", "confidence": round(1.0 - avg_score, 4)}


# ──────────────────────────────────────────────────────────────────────────────
# Text analysis (metadata)
# ──────────────────────────────────────────────────────────────────────────────
def analyze_text(metadata: dict) -> dict:
    """
    Analyzes video title and description for keywords indicating AI-generated content.
    Returns a score from 0.0 to 1.0 (1.0 = definitely AI generated).
    """
    title = metadata.get("title", "").lower()
    desc = metadata.get("description", "").lower()
    full_text = f"{title} {desc}"

    # High-confidence indicators (often explicitly stated)
    ai_keywords = [
        "ai generated", "ai-generated", "ai video", "synthetic media", 
        "generated with ai", "created with ai", "sora", "runway gen", 
        "stable video diffusion", "pika labs", "kling ai", "luma dream machine",
        "flux ai", "midjourney video"
    ]
    
    # Check for keywords
    matches = [kw for kw in ai_keywords if kw in full_text]
    
    if matches:
        logger.info(f"AI keywords found in metadata: {matches}")
        return {"score": 0.95, "reason": f"AI keyword(s) found: {', '.join(matches)}"}
    
    # Middle indicators (may be AI or just tech-related)
    soft_keywords = ["ai", "prompt", "artificial intelligence", "generative"]
    soft_matches = [kw for kw in soft_keywords if kw in full_text]
    if soft_matches:
        return {"score": 0.4, "reason": "Weak AI context found"}

    return {"score": 0.0, "reason": "No AI keywords found"}


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────
def analyze_frame(b64_image: str, metadata: dict | None = None) -> dict:
    """
    Optimized analysis pipeline for speed.
    """
    if metadata is None:
        metadata = {}

    try:
        img = decode_image(b64_image)
        
        with torch.inference_mode():
            # ── Step 1: Text analysis (Metadata) ──────────────────────────────────
            text_result = analyze_text(metadata)
            text_score = text_result["score"]

            # ── Step 2: CLIP scene classification ─────────────────────────────────
            clip_result = classify_scene(img)
            logger.info(f"CLIP result: {clip_result}")

            # ── Step 3: Face detection & Deepfake Analysis ────────────────────────
            faces = detect_faces(img)
            logger.info(f"Detected {len(faces)} face(s) in frame.")

            if faces:
                deepfake_result = detect_deepfake(faces)
                # If deepfake is detected, it usually overrides other's labels
                if deepfake_result["type"] == "deepfake_detected":
                    return deepfake_result

            # ── Step 4: Weighted Signal Aggregation ───────────────────────────────
            # Weights: 
            # - Visual (CLIP) provides the base classification.
            # - Text (Metadata) is a strong bias if keywords are found.

            final_type = clip_result["type"]
            final_conf = clip_result["confidence"]

            # If text analysis is very confident (0.95), and visual is "real" or "ai"
            if text_score > 0.90:
                if clip_result["type"] in ["real_video", "ai_generated"]:
                    final_type = "ai_generated"
                    # Weighted boost: 70% Text, 30% Visual
                    final_conf = (0.7 * text_score) + (0.3 * clip_result["confidence"])
            
            # If CLIP says it's AI, and text doesn't contradict it (score >= 0)
            elif clip_result["type"] == "ai_generated" and clip_result["confidence"] > 0.6:
                # Neutral boost if soft keywords are present
                if text_score > 0.3:
                    final_conf = min(0.99, final_conf + 0.1)

            # Safety threshold: if visual is "real" but text is very AI-heavy
            if final_type == "real_video" and text_score > 0.90:
                # Override to AI Generated but with lower confidence (unclear case)
                final_type = "ai_generated"
                final_conf = 0.7

            return {
                "type": final_type,
                "confidence": round(float(final_conf), 4)
            }

    except Exception as exc:
        logger.exception("Frame analysis failed")
        return {"type": "error", "confidence": 0.0, "detail": str(exc)}
