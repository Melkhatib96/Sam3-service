"""
Main FastAPI application for Sam3 Segmentation Service.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.model import SAM3Segmenter, init_segmenter

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)

# How often the watchdog wakes up to check idle time (seconds)
_WATCHDOG_INTERVAL = 5 * 60  # 5 minutes


async def _model_watchdog(segmenter: SAM3Segmenter) -> None:
    """
    Background task: unloads the model from RAM when it has been idle for
    at least SAM3Segmenter.IDLE_TIMEOUT_SECONDS (30 min).

    Wakes every _WATCHDOG_INTERVAL seconds (5 min) to check.
    The next segment() call after an unload triggers an automatic reload
    from the local disk file — no S3 round-trip.
    """
    idle_minutes = SAM3Segmenter.IDLE_TIMEOUT_SECONDS // 60
    idle_seconds = SAM3Segmenter.IDLE_TIMEOUT_SECONDS % 60
    idle_str = f"{idle_minutes} min" if idle_seconds == 0 else f"{SAM3Segmenter.IDLE_TIMEOUT_SECONDS}s"
    check_str = f"{_WATCHDOG_INTERVAL}s" if _WATCHDOG_INTERVAL < 60 else f"{_WATCHDOG_INTERVAL // 60} min"
    logger.info(
        "Model watchdog started — will unload after %s of inactivity (checking every %s).",
        idle_str,
        check_str,
    )
    try:
        while True:
            await asyncio.sleep(_WATCHDOG_INTERVAL)

            if not segmenter.is_loaded:
                continue

            if segmenter.is_idle:
                logger.info(
                    "SAM3 model has been idle for %.1f min — unloading from RAM.",
                    (SAM3Segmenter.IDLE_TIMEOUT_SECONDS) / 60,
                )
                # unload() is fast (just sets None + gc), but run in executor
                # to avoid any GC hiccup blocking the event loop.
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, segmenter.unload)

    except asyncio.CancelledError:
        logger.info("Model watchdog stopped.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    App lifespan:
    - Startup  → download model from S3 (if not cached) and load into RAM.
                 Start the idle-unload watchdog task.
    - Shutdown → cancel watchdog.
    """
    logger.info(
        "Starting %s v%s  [%s]", settings.APP_NAME, settings.APP_VERSION, settings.ENVIRONMENT
    )
    logger.info("Model path : %s", settings.MODEL_LOCAL_PATH)
    logger.info("S3 bucket  : %s  key: %s", settings.S3_BUCKET, settings.S3_MODEL_KEY)

    segmenter = None
    watchdog_task = None

    try:
        segmenter = init_segmenter()
        logger.info("SAM3 model ready.")
    except Exception as exc:
        # Log but do NOT crash — Railway will restart after the healthcheck fails.
        logger.error("Model initialization failed: %s", exc, exc_info=True)

    if segmenter is not None:
        watchdog_task = asyncio.create_task(_model_watchdog(segmenter))

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    if watchdog_task is not None:
        watchdog_task.cancel()
        try:
            await watchdog_task
        except asyncio.CancelledError:
            pass

    logger.info("Shutting down %s", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Microservice for text-guided image segmentation using SAM3. "
        "Accepts an image (URL or base64) and a comma-separated list of class names; "
        "returns detected objects with confidence scores and bounding boxes."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.cors_allow_methods_list,
    allow_headers=settings.cors_allow_headers_list,
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "segment": "/api/v1/segment",
            "health": "/api/v1/health",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "HTTPException", "message": exc.detail, "status_code": exc.status_code},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": "ValidationError", "message": "Request validation failed", "details": exc.errors()},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": str(exc) if settings.DEBUG else "An unexpected error occurred.",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
