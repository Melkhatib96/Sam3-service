"""
SAM3 model loader and inference engine.

Download flow (once at startup):
  1. Check if MODEL_LOCAL_PATH already exists (cached from a previous run).
  2. If not, stream-download from the Railway S3 bucket using boto3.

Inference:
  Uses SAM3SemanticPredictor — the correct ultralytics interface for
  text-prompted concept segmentation introduced in ultralytics 8.3.237.

  Call flow per request:
    predictor.set_image(image)       # encodes image features
    results = predictor(text=classes) # runs detector head with text prompts

  Because SAM3SemanticPredictor stores image state internally, the lock is
  held for the ENTIRE set_image → predict cycle so concurrent requests are
  serialised.  This is appropriate: at 3.3 GB the model cannot run two
  inferences in parallel on a memory-constrained host anyway.

Spooling / idle-unload:
  After IDLE_TIMEOUT_SECONDS (default 30 min) of no inference the watchdog
  (started by main.py) calls unload(), freeing RAM.  The next segment()
  call reloads from the local disk file — no S3 round-trip.
"""
from __future__ import annotations

import gc
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image

from app.core.config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Detection:
    """A single detected / segmented object."""
    class_name: str
    confidence: float
    bbox: List[float]           # [x1, y1, x2, y2] in pixels
    mask_area: Optional[int] = None  # pixel count of the segmentation mask


# ─────────────────────────────────────────────────────────────────────────────
# S3 download
# ─────────────────────────────────────────────────────────────────────────────

def download_model_from_s3() -> None:
    """
    Stream the model weights from the Railway S3 bucket to MODEL_LOCAL_PATH.
    Safe to call if the file already exists (skips download).
    """
    local_path = Path(settings.MODEL_LOCAL_PATH)

    if local_path.exists():
        logger.info("Model already cached at %s — skipping download.", local_path)
        return

    local_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Downloading model from s3://%s/%s → %s …",
        settings.AWS_S3_BUCKET_NAME, settings.S3_MODEL_KEY, local_path,
    )

    s3 = boto3.client(
        "s3",
        endpoint_url=settings.AWS_ENDPOINT_URL,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        region_name=settings.AWS_DEFAULT_REGION,
    )

    try:
        meta = s3.head_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=settings.S3_MODEL_KEY)
        total_bytes = meta["ContentLength"]
        logger.info("Model size: %.1f MB", total_bytes / 1024 / 1024)
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(
            f"Cannot access model in S3 bucket '{settings.AWS_S3_BUCKET_NAME}': {exc}"
        ) from exc

    chunk_size = 64 * 1024 * 1024  # 64 MB
    downloaded = 0
    tmp_path = local_path.with_suffix(".tmp")

    try:
        response = s3.get_object(Bucket=settings.AWS_S3_BUCKET_NAME, Key=settings.S3_MODEL_KEY)
        body = response["Body"]

        with open(tmp_path, "wb") as fh:
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / total_bytes * 100
                logger.info(
                    "Download progress: %.1f%%  (%.0f / %.0f MB)",
                    pct, downloaded / 1024 / 1024, total_bytes / 1024 / 1024,
                )

        tmp_path.rename(local_path)
        logger.info("Model saved to %s", local_path)

    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"Model download failed: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SAM3Segmenter:
    """
    Wraps SAM3SemanticPredictor with a text-prompt segmentation interface
    and built-in idle-unload / lazy-reload (spooling) support.

    SAM3SemanticPredictor is the correct ultralytics interface for SAM3's
    Promptable Concept Segmentation (PCS) feature — it accepts a list of
    text class names and returns all matching instances with masks and scores.
    """

    IDLE_TIMEOUT_SECONDS: int = 30 * 60  # 30 minutes

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._predictor = None   # SAM3SemanticPredictor instance
        self._last_used: float = 0.0   # time.monotonic() timestamp
        self._lock = threading.Lock()

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def is_idle(self) -> bool:
        return self._last_used > 0.0 and (time.monotonic() - self._last_used) >= self.IDLE_TIMEOUT_SECONDS

    # ── Load / unload ─────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load SAM3 weights from disk into RAM. Thread-safe; no-op if already loaded."""
        with self._lock:
            if self._predictor is not None:
                return
            self._load_locked()

    def _load_locked(self) -> None:
        """Internal load — must be called while self._lock is held."""
        from ultralytics.models.sam import SAM3SemanticPredictor  # noqa: PLC0415

        logger.info("Loading SAM3 SemanticPredictor from %s …", self._model_path)
        overrides = dict(
            conf=settings.CONFIDENCE_THRESHOLD,
            task="segment",
            mode="predict",
            model=self._model_path,
            verbose=False,
            save=False,
        )
        self._predictor = SAM3SemanticPredictor(overrides=overrides)
        self._last_used = time.monotonic()  # idle timer starts from load time
        logger.info("SAM3 loaded successfully.")

    def unload(self) -> None:
        """
        Release SAM3 weights from RAM.
        Called by the idle watchdog; the next segment() call reloads from disk.
        """
        with self._lock:
            if self._predictor is None:
                return
            logger.info(
                "Unloading SAM3 from RAM (idle for %.1f min).",
                (time.monotonic() - self._last_used) / 60,
            )
            self._predictor = None
            self._last_used = 0.0

        try:
            import torch  # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()

        # Force the C allocator to return free pages to the OS.
        # gc.collect() removes Python references but PyTorch's CPU memory
        # allocator holds onto pages in its own pool — malloc_trim flushes them.
        # This is a no-op on non-Linux platforms (Windows/macOS).
        try:
            import ctypes  # noqa: PLC0415
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            logger.info("malloc_trim called — free pages returned to OS.")
        except Exception:
            pass

        logger.info("SAM3 unloaded.")

    # ── Inference ─────────────────────────────────────────────────────────────

    def segment(self, image: Image.Image, classes: List[str]) -> List[Detection]:
        """
        Run text-guided concept segmentation on *image*.

        The lock is held for the FULL set_image → predict cycle because
        SAM3SemanticPredictor stores image state internally (set_image encodes
        image features into the predictor).  Concurrent requests are serialised;
        this is acceptable given the model's memory footprint.

        Args:
            image:   PIL Image in RGB mode (guaranteed by _load_image in segment.py).
            classes: Class names to find, e.g. ["car", "wheel", "window"].

        Returns:
            Detections sorted by confidence (desc), already filtered by the
            predictor's conf threshold (set in overrides at init time).
        """
        with self._lock:
            if self._predictor is None:
                logger.info("SAM3 not in RAM — reloading from disk …")
                self._load_locked()

            self._last_used = time.monotonic()
            detections = self._run_inference(self._predictor, image, classes)

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ── SAM3 inference ────────────────────────────────────────────────────────

    @staticmethod
    def _run_inference(
        predictor,
        image: Image.Image,
        classes: List[str],
    ) -> List[Detection]:
        """
        Call SAM3SemanticPredictor with text class prompts and parse results.

        SAM3's Promptable Concept Segmentation (PCS) finds ALL instances of
        every requested concept in the image, returning per-instance confidence
        scores, bounding boxes, and segmentation masks.

        The `text` parameter takes a list of noun-phrase strings; class indices
        in the results map back to positions in that list.
        """
        # Encode image features then run concept detection — both calls are
        # the official SAM3SemanticPredictor API, no extra pre-processing.
        predictor.set_image(image)
        results = predictor(text=classes)

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            # names is a dict mapping cls_id → class_name string.
            # For SAM3 text prompts, cls_id maps to the index in the `classes` list.
            names = result.names

            for i in range(len(result.boxes)):
                cls_id = int(result.boxes.cls[i].item())
                confidence = float(result.boxes.conf[i].item())
                bbox = result.boxes.xyxy[i].cpu().numpy().tolist()

                # Resolve class name.
                # SAM3SemanticPredictor returns names as a list (e.g. ["car", "wheel"]).
                # Standard YOLO models return a dict ({0: "car", 1: "wheel"}).
                if isinstance(names, dict):
                    class_name = names.get(
                        cls_id,
                        classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}",
                    )
                else:  # list
                    class_name = (
                        names[cls_id] if cls_id < len(names)
                        else classes[cls_id] if cls_id < len(classes)
                        else f"class_{cls_id}"
                    )

                mask_area: Optional[int] = None
                if result.masks is not None and i < len(result.masks.data):
                    mask_area = int(result.masks.data[i].sum().item())

                detections.append(
                    Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=bbox,
                        mask_area=mask_area,
                    )
                )

        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_segmenter: Optional[SAM3Segmenter] = None


def get_segmenter() -> SAM3Segmenter:
    """Return the global SAM3Segmenter singleton."""
    if _segmenter is None:
        raise RuntimeError("Segmenter not initialised. Call init_segmenter() first.")
    return _segmenter


def init_segmenter() -> SAM3Segmenter:
    """
    Download model from S3 (if not cached locally), load into RAM, register singleton.
    Called once during app lifespan startup for warmup.
    Subsequent reloads after idle-unload are handled transparently by segment().
    """
    global _segmenter
    download_model_from_s3()
    segmenter = SAM3Segmenter(settings.MODEL_LOCAL_PATH)
    segmenter.load()
    _segmenter = segmenter
    return _segmenter
