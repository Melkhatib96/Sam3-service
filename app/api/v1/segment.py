"""
Segmentation endpoint — the primary API for this service.

POST /api/v1/segment
  Body : SegmentRequest  { image: url_or_base64, prompt: "car, wheel" }
  Reply: SegmentResponse { detections: [...], total_detections, processing_time }
"""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import time
from typing import List

import httpx
from fastapi import APIRouter, Depends, HTTPException, status
from PIL import Image

from app.core.auth import get_current_user_from_api_key
from app.core.config import settings
from app.core.model import get_segmenter
from app.db.models import User
from app.models.segment import DetectionResult, SegmentRequest, SegmentResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# Re-used async HTTP client for URL downloads
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
    return _http_client


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _load_image(image_str: str) -> Image.Image:
    """
    Accept either:
      • An HTTP/HTTPS URL  → download with httpx
      • A base64 string    → decode directly (with or without the data-URI prefix)

    Returns a PIL Image in RGB mode.
    Raises HTTPException(400) on bad input.
    """
    if image_str.startswith("http://") or image_str.startswith("https://"):
        try:
            resp = await _get_http_client().get(image_str)
            resp.raise_for_status()
            data = resp.content
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to download image from URL: {exc}",
            ) from exc
    else:
        # Strip the optional data-URI prefix
        if "," in image_str:
            image_str = image_str.split(",", 1)[1]
        try:
            data = base64.b64decode(image_str)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 image string: {exc}",
            ) from exc

    # Validate size
    if len(data) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {settings.MAX_IMAGE_SIZE_MB} MB limit.",
        )

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot decode image: {exc}",
        ) from exc

    return image


def _parse_classes(prompt: str) -> List[str]:
    """Split the comma-separated prompt into a cleaned list of class names."""
    classes = [c.strip() for c in prompt.split(",") if c.strip()]
    if not classes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt must contain at least one class name.",
        )
    return classes


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/segment",
    response_model=SegmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Segment objects in an image by text class prompt",
)
async def segment_image(
    body: SegmentRequest,
    current_user: User = Depends(get_current_user_from_api_key),
) -> SegmentResponse:
    """
    Detect and segment objects in an image using SAM3.

    - **image**: public URL *or* base64-encoded image string
    - **prompt**: comma-separated class names, e.g. ``"car, wheel, window"``

    Returns a list of detected objects with their class, confidence score,
    bounding box, and (when available) segmentation mask area in pixels.
    """
    t0 = time.perf_counter()
    classes = _parse_classes(body.prompt)
    logger.info(
        "Segment request — user='%s'  classes=%s  image='%s…'",
        current_user.username,
        classes,
        body.image[:60],
    )

    image = await _load_image(body.image)

    try:
        segmenter = get_segmenter()
        # segment() is CPU-bound and may trigger a disk reload after an idle-unload.
        # Run it in a thread-pool executor so the event loop stays responsive.
        loop = asyncio.get_event_loop()
        raw_detections = await loop.run_in_executor(
            None, segmenter.segment, image, classes
        )
    except RuntimeError as exc:
        logger.error("Model inference error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not ready: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Segmentation failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation failed: {exc}",
        ) from exc

    detections = [
        DetectionResult(
            class_name=d.class_name,
            confidence=round(d.confidence, 4),
            bbox=[round(v, 2) for v in d.bbox],
            mask_area=d.mask_area,
        )
        for d in raw_detections
    ]

    processing_time = round(time.perf_counter() - t0, 3)
    logger.info(
        "Segment done — user='%s'  found=%d  time=%.3fs",
        current_user.username,
        len(detections),
        processing_time,
    )

    return SegmentResponse(
        success=True,
        total_detections=len(detections),
        detections=detections,
        processing_time=processing_time,
    )
