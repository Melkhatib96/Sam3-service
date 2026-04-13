"""
SAM3 model loader and inference engine.

Download flow (once at startup):
  1. Check if MODEL_LOCAL_PATH already exists (cached from a previous run).
  2. If not, stream-download from the Railway S3 bucket using boto3.

Inference — single worker thread design:
  A single dedicated thread owns the predictor for its entire lifetime.
  Async requests submit jobs to a queue and `await` an asyncio.Future for
  the result.  While waiting, the coroutine is suspended (zero CPU usage),
  so PyTorch's internal BLAS thread pool keeps all vCPUs for itself.

  Under the old threading.Lock design, N concurrent requests spawned N
  executor threads that all competed with PyTorch's BLAS threads for the
  same vCPUs, making each inference 5-10× slower under concurrency.

  Call flow per request:
    predictor.set_image(image)        # encodes image features (ViT forward)
    results = predictor(text=classes) # runs detector head + mask decoder

Spooling / idle-unload:
  After IDLE_TIMEOUT_SECONDS (default 30 min) of no inference the watchdog
  (started by main.py) calls unload(), which enqueues an unload sentinel.
  The worker thread processes it and frees RAM.  The next segment() call
  re-enqueues a load + inference job — no S3 round-trip.
"""
from __future__ import annotations

import asyncio
import gc
import logging
import queue as _queue
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
# Worker-thread helpers
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel objects placed on the queue to signal control actions.
_STOP   = object()  # shut down the worker thread
_UNLOAD = object()  # unload the model from RAM


@dataclass
class _Job:
    """An inference job submitted by an async request to the worker thread."""
    image:   Image.Image
    classes: List[str]
    loop:    asyncio.AbstractEventLoop
    future:  "asyncio.Future[List[Detection]]"


def _resolve(future: asyncio.Future, result: object) -> None:
    """Set result on a Future that may already be cancelled — safe no-op."""
    if not future.done():
        future.set_result(result)


def _reject(future: asyncio.Future, exc: BaseException) -> None:
    """Set exception on a Future that may already be cancelled — safe no-op."""
    if not future.done():
        future.set_exception(exc)


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
    Wraps SAM3SemanticPredictor with a single-worker-thread inference queue.

    Architecture
    ─────────────
    One dedicated daemon thread (`sam3-worker`) owns the predictor object for
    its entire lifetime.  Async callers submit _Job items to a SimpleQueue and
    suspend on an asyncio.Future; the worker resolves each Future when
    inference completes.

    This guarantees that PyTorch's internal BLAS/OpenMP thread pool always
    has full access to every vCPU, regardless of how many HTTP requests are
    in flight simultaneously.

    Lifecycle
    ─────────
    1. download_model_from_s3() / init_segmenter()  — before any requests
    2. load()   — loads weights into RAM (called in main thread at startup)
    3. start()  — starts the worker thread (called right after load())
    4. [requests flow through segment()]
    5. stop()   — sends stop sentinel; called at app shutdown
    """

    IDLE_TIMEOUT_SECONDS: int = 30 * 60  # 30 minutes

    def __init__(self, model_path: str):
        self._model_path = model_path
        self._predictor   = None          # SAM3SemanticPredictor; owned by worker thread
        self._last_used: float = 0.0      # monotonic timestamp; written only by worker
        self._queue: _queue.SimpleQueue = _queue.SimpleQueue()
        self._worker: Optional[threading.Thread] = None

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def is_idle(self) -> bool:
        return (
            self._last_used > 0.0
            and (time.monotonic() - self._last_used) >= self.IDLE_TIMEOUT_SECONDS
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def load(self) -> None:
        """
        Load SAM3 weights from disk into RAM.

        Must be called BEFORE start() — runs synchronously in the calling
        thread (no worker thread yet).  No-op if already loaded.
        """
        if self._predictor is None:
            self._load()

    def start(self) -> None:
        """Start the single inference worker thread. Call once after load()."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="sam3-worker"
        )
        self._worker.start()
        logger.info("SAM3 inference worker thread started.")

    def stop(self) -> None:
        """
        Gracefully stop the worker thread.  Blocks up to 5 s for the thread
        to finish its current job.  Call this during app shutdown.
        """
        self._queue.put(_STOP)
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None

    def unload(self) -> None:
        """
        Schedule a model unload via the worker thread.

        Non-blocking: enqueues a sentinel and returns immediately.
        The worker releases the predictor and calls gc / malloc_trim.
        The next segment() call after an unload triggers automatic reload
        from the local disk file — no S3 round-trip.
        """
        self._queue.put(_UNLOAD)

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _worker_loop(self) -> None:
        """
        Main loop of the single inference worker thread.

        Blocks on SimpleQueue.get() (zero CPU while idle) and processes
        one job at a time:
          • _Job      → run inference, resolve future
          • _UNLOAD   → release predictor from RAM
          • _STOP     → exit the loop
        """
        logger.info("SAM3 worker: ready for jobs.")
        while True:
            job = self._queue.get()  # blocks; no busy-wait

            if job is _STOP:
                logger.info("SAM3 worker: stop sentinel — exiting.")
                break

            if job is _UNLOAD:
                if self._predictor is not None:
                    self._do_unload()
                continue

            # Inference job
            assert isinstance(job, _Job)
            try:
                if self._predictor is None:
                    logger.info("SAM3 not in RAM — reloading from disk …")
                    self._load()

                self._last_used = time.monotonic()
                detections = self._run_inference(self._predictor, job.image, job.classes)
                detections.sort(key=lambda d: d.confidence, reverse=True)
                job.loop.call_soon_threadsafe(_resolve, job.future, detections)

            except Exception as exc:
                logger.error("SAM3 worker: inference error: %s", exc, exc_info=True)
                job.loop.call_soon_threadsafe(_reject, job.future, exc)

    # ── Load / unload (called only from worker thread or during startup) ──────

    def _load(self) -> None:
        """Load SAM3 weights. Called from worker thread or main thread at startup."""
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
        self._last_used = time.monotonic()
        logger.info("SAM3 loaded successfully.")

    def _do_unload(self) -> None:
        """Release predictor from RAM. Called only from the worker thread."""
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
        # allocator holds onto pages — malloc_trim flushes them.
        # No-op on non-Linux platforms.
        try:
            import ctypes  # noqa: PLC0415
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            logger.info("malloc_trim called — free pages returned to OS.")
        except Exception:
            pass

        logger.info("SAM3 unloaded.")

    # ── Public async inference API ────────────────────────────────────────────

    async def segment(self, image: Image.Image, classes: List[str]) -> List[Detection]:
        """
        Submit an inference job to the worker thread and await the result.

        The calling coroutine suspends (zero CPU) while the worker runs
        inference.  PyTorch therefore has exclusive access to all vCPUs,
        keeping single-request latency constant regardless of concurrency.
        """
        loop   = asyncio.get_running_loop()
        future: asyncio.Future[List[Detection]] = loop.create_future()
        self._queue.put(_Job(image=image, classes=classes, loop=loop, future=future))
        return await future

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
        predictor.set_image(image)
        results = predictor(text=classes)

        detections: List[Detection] = []

        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue

            names = result.names

            for i in range(len(result.boxes)):
                cls_id     = int(result.boxes.cls[i].item())
                confidence = float(result.boxes.conf[i].item())
                bbox       = result.boxes.xyxy[i].cpu().numpy().tolist()

                # SAM3SemanticPredictor returns names as a list; standard YOLO
                # models return a dict ({0: "car"}).  Handle both.
                if isinstance(names, dict):
                    class_name = names.get(
                        cls_id,
                        classes[cls_id] if cls_id < len(classes) else f"class_{cls_id}",
                    )
                else:  # list
                    class_name = (
                        names[cls_id]   if cls_id < len(names)
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
    Download model from S3 (if not cached locally), load into RAM, start the
    worker thread, and register the singleton.

    Called once during app lifespan startup.
    Startup sequence: download → load (main thread) → start worker thread.
    """
    global _segmenter
    download_model_from_s3()
    segmenter = SAM3Segmenter(settings.MODEL_LOCAL_PATH)
    segmenter.load()   # load weights synchronously before worker starts
    segmenter.start()  # worker now owns the predictor for all future requests
    _segmenter = segmenter
    return _segmenter
