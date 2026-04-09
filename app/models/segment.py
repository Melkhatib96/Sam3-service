"""
Pydantic request / response models for the segmentation API.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class SegmentRequest(BaseModel):
    """
    Request body for POST /api/v1/segment.

    Provide the image as either a public URL **or** a base64-encoded string
    (with or without the ``data:image/…;base64,`` prefix).
    """
    image: str = Field(
        ...,
        description="Public image URL (http/https) OR base64-encoded image string.",
        examples=["https://example.com/car.jpg"],
    )
    prompt: str = Field(
        ...,
        description="Comma-separated class names to detect, e.g. 'car, wheel, window'.",
        examples=["car, wheel, window"],
    )


class DetectionResult(BaseModel):
    """A single detected / segmented object."""
    class_name: str = Field(..., description="Detected class label.")
    confidence: float = Field(..., description="Confidence / IoU score (0–1).", ge=0, le=1)
    bbox: List[float] = Field(
        ...,
        description="Bounding box [x1, y1, x2, y2] in pixels.",
        min_length=4,
        max_length=4,
    )
    mask_area: Optional[int] = Field(
        None,
        description="Pixel count of the segmentation mask (null if model is detection-only).",
    )


class SegmentResponse(BaseModel):
    """Response for POST /api/v1/segment."""
    success: bool = Field(default=True)
    total_detections: int = Field(..., description="Number of objects found.")
    detections: List[DetectionResult] = Field(
        ...,
        description="List of detected objects, sorted by confidence (highest first).",
    )
    processing_time: float = Field(..., description="Inference wall-clock time in seconds.")
