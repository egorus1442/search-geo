import time
import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from config import get_logger, get_settings
from services.api.schemas import LocalizeResponse
from services.matching.localize import localize

router = APIRouter(prefix="/api/v1", tags=["localize"])
logger = get_logger(__name__)
_s = get_settings()

_MAX_BYTES = _s.max_image_size_mb * 1024 * 1024


@router.post("/localize", response_model=LocalizeResponse)
async def localize_image(
    image: UploadFile = File(..., description="Фото с дрона/зонда (JPEG/PNG)"),
    top_n: int = Form(default=10, ge=1, le=50),
):
    """
    Геолокализация изображения по эталонной базе.

    Возвращает top-N кандидатов с координатами и оценкой уверенности.
    """
    task_id = str(uuid.uuid4())

    if image.content_type not in ("image/jpeg", "image/png", "image/tiff"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Supported formats: JPEG, PNG, TIFF",
        )

    image_bytes = await image.read()
    if len(image_bytes) > _MAX_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large. Max size: {_s.max_image_size_mb} MB",
        )

    logger.info("localize_request", task_id=task_id, size_kb=len(image_bytes) // 1024)
    t0 = time.monotonic()

    try:
        candidates = localize(image_bytes, top_n=top_n)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Index not ready: {exc}",
        )
    except Exception as exc:
        logger.error("localize_error", task_id=task_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Localization failed",
        )

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    logger.info("localize_done", task_id=task_id, elapsed_ms=elapsed_ms, n=len(candidates))

    return LocalizeResponse(
        task_id=task_id,
        status="completed",
        processing_time_ms=elapsed_ms,
        candidates=candidates,
    )
