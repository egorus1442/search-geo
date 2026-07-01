"""MinIO / S3 хранилище для тайлов и патчей."""
from io import BytesIO
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from config import get_logger, get_settings

logger = get_logger(__name__)
_settings = get_settings()


def _get_client() -> Minio:
    return Minio(
        endpoint=_settings.minio_endpoint,
        access_key=_settings.minio_access_key,
        secret_key=_settings.minio_secret_key,
        secure=_settings.minio_secure,
    )


def ensure_bucket(bucket: str | None = None) -> None:
    """Создать bucket если не существует."""
    bucket = bucket or _settings.minio_bucket
    client = _get_client()
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logger.info("minio_bucket_created", bucket=bucket)


def upload_file(local_path: Path, s3_key: str, bucket: str | None = None) -> str:
    """Загрузить файл в MinIO. Возвращает s3_key."""
    bucket = bucket or _settings.minio_bucket
    client = _get_client()
    client.fput_object(bucket, s3_key, str(local_path))
    logger.debug("minio_upload", key=s3_key, size_bytes=local_path.stat().st_size)
    return s3_key


def upload_bytes(data: bytes, s3_key: str, content_type: str = "image/png", bucket: str | None = None) -> str:
    """Загрузить bytes-объект в MinIO."""
    bucket = bucket or _settings.minio_bucket
    client = _get_client()
    client.put_object(
        bucket,
        s3_key,
        BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return s3_key


def download_bytes(s3_key: str, bucket: str | None = None) -> bytes:
    """Скачать объект из MinIO в bytes."""
    bucket = bucket or _settings.minio_bucket
    client = _get_client()
    resp = client.get_object(bucket, s3_key)
    try:
        return resp.read()
    finally:
        resp.close()
        resp.release_conn()


def get_presigned_url(s3_key: str, bucket: str | None = None, expires_seconds: int = 3600) -> str:
    """Получить временную ссылку на объект."""
    from datetime import timedelta
    bucket = bucket or _settings.minio_bucket
    client = _get_client()
    return client.presigned_get_object(bucket, s3_key, expires=timedelta(seconds=expires_seconds))


def object_exists(s3_key: str, bucket: str | None = None) -> bool:
    bucket = bucket or _settings.minio_bucket
    client = _get_client()
    try:
        client.stat_object(bucket, s3_key)
        return True
    except S3Error:
        return False
