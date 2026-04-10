"""
Configuration for Sam3 Segmentation Service.
All settings are loaded from environment variables / .env file.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    # ── Application ───────────────────────────────────────────
    APP_NAME: str = "Sam3 Segmentation Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # ── Server ────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8008

    # ── Database (shared PostgreSQL with gateway for API-key auth) ──
    DATABASE_URL: str  # required

    # ── CORS ──────────────────────────────────────────────────
    CORS_ORIGINS: str  # required, comma-separated
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: str = "*"
    CORS_ALLOW_HEADERS: str = "*"

    # ── Model / S3 bucket ─────────────────────────────────────
    AWS_ENDPOINT_URL: str       # required - e.g. https://t3.storageapi.dev
    AWS_S3_BUCKET_NAME: str     # required - e.g. indexed-stashbox-ehdhwgm7
    AWS_DEFAULT_REGION: str = "auto"
    AWS_ACCESS_KEY_ID: str      # required
    AWS_SECRET_ACCESS_KEY: str  # required
    S3_MODEL_KEY: str = "models/sam3.pt"

    # Local path where the model is cached after download from S3
    MODEL_LOCAL_PATH: str = "/app/models/sam3.pt"

    # Detections below this confidence are filtered out
    CONFIDENCE_THRESHOLD: float = 0.25

    # Max uploaded image size in MB
    MAX_IMAGE_SIZE_MB: int = 20

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        if isinstance(self.CORS_ORIGINS, str):
            return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        return self.CORS_ORIGINS

    @property
    def cors_allow_methods_list(self) -> List[str]:
        if self.CORS_ALLOW_METHODS == "*":
            return ["*"]
        return [m.strip() for m in self.CORS_ALLOW_METHODS.split(",") if m.strip()]

    @property
    def cors_allow_headers_list(self) -> List[str]:
        if self.CORS_ALLOW_HEADERS == "*":
            return ["*"]
        return [h.strip() for h in self.CORS_ALLOW_HEADERS.split(",") if h.strip()]

    @property
    def max_image_size_bytes(self) -> int:
        return self.MAX_IMAGE_SIZE_MB * 1024 * 1024


settings = Settings()
