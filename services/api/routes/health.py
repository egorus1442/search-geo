from fastapi import APIRouter
from sqlalchemy import text

from services.api.schemas import HealthResponse
from services.db.session import AsyncSessionLocal
from config import get_settings

router = APIRouter(tags=["health"])
_s = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    components: dict[str, str] = {}

    # PostgreSQL
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        components["postgres"] = "ok"
    except Exception as exc:
        components["postgres"] = f"error: {exc}"

    # Redis
    try:
        import redis as redis_lib
        r = redis_lib.from_url(_s.redis_url, socket_connect_timeout=2)
        r.ping()
        components["redis"] = "ok"
    except Exception as exc:
        components["redis"] = f"error: {exc}"

    # MinIO
    try:
        from minio import Minio
        client = Minio(
            _s.minio_endpoint,
            access_key=_s.minio_access_key,
            secret_key=_s.minio_secret_key,
            secure=_s.minio_secure,
        )
        client.bucket_exists(_s.minio_bucket)
        components["minio"] = "ok"
    except Exception as exc:
        components["minio"] = f"error: {exc}"

    # FAISS index
    try:
        from services.matching.localize import _get_faiss
        store = _get_faiss()
        components["faiss"] = f"ok (ntotal={store.ntotal})"
    except Exception as exc:
        components["faiss"] = f"not_loaded: {exc}"

    status = "ok" if all(v == "ok" or v.startswith("ok") for v in components.values()) else "degraded"
    return HealthResponse(status=status, components=components)
