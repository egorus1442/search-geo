"""
Celery task: скачать мозаику Esri World Imagery по bbox и нарезать на патчи.

Источник эталонной базы переведён с Sentinel-2/CDSE на Esri World Imagery
(~0.5–1 м/пкс вместо 10 м) — это снимает первопричину провала verifier'а на
низкодетальной местности. Сам CDSE-клиент (services/ingestor/cdse_client.py)
оставлен в репозитории для справки, но в пайплайне ingestion не используется.

Очередь: ingest
"""
import tempfile
from pathlib import Path

from workers.celery_app import app
from config import get_logger, get_settings
from services.db.models import utcnow
from services.db.session import SyncSessionLocal
from services.index.metadata_store import PatchRepo
from services.ingestor.esri_client import fetch_mosaic_to_geotiff
from services.ingestor.tile_cutter import cut_patches_from_raster
from services.ingestor.storage import ensure_bucket

logger = get_logger(__name__)
_s = get_settings()


@app.task(
    bind=True,
    name="workers.tasks.ingest_task.run_ingest",
    queue="ingest",
    max_retries=2,
    default_retry_delay=60,
)
def run_ingest(
    self,
    bbox: list[float],
    gsd_m: float | None = None,
    patch_size: int | None = None,
    overlap_ratio: float | None = None,
    task_db_id: str | None = None,
    max_patches: int | None = None,
    clip_to_bbox: bool = False,
    run_label: str | None = None,
) -> dict:
    """
    Скачать мозаику Esri World Imagery на bbox и нарезать на патчи.

    bbox: [lon_min, lat_min, lon_max, lat_max] (WGS84)
    gsd_m: целевое разрешение эталона, м/пкс (по умолчанию settings.esri_gsd_m)
    patch_size: размер патча в пикселях (footprint_м = patch_size * gsd_m)
    run_label: метка источника; нужна для повторной нарезки того же bbox
    """
    gsd_m = gsd_m or _s.esri_gsd_m
    logger.info("ingest_start", bbox=bbox, gsd_m=gsd_m, patch_size=patch_size)

    ensure_bucket()

    # Уникальный идентификатор источника (unique-constraint в SourceTile).
    # Повторную нарезку того же bbox разрешаем через run_label.
    lon_min, lat_min, lon_max, lat_max = bbox
    source_id = (
        f"esri:{lon_min:.4f},{lat_min:.4f},{lon_max:.4f},{lat_max:.4f}@{gsd_m}"
    )
    if run_label:
        source_id = f"{source_id}:{run_label}"

    stats = {"patches_created": 0, "skipped": 0, "source_id": source_id}

    with SyncSessionLocal() as session:
        repo = PatchRepo(session)

        if repo.tile_exists(source_id):
            logger.info("ingest_skip_existing", source_id=source_id)
            stats["skipped"] = 1
            return stats

        tile_record = repo.create_source_tile(
            product_id=source_id,
            bbox=(lon_min, lat_min, lon_max, lat_max),
            date_acq=utcnow(),
            cloud_cover=None,
        )
        session.commit()

        with tempfile.TemporaryDirectory() as tmpdir:
            tif_path = Path(tmpdir) / "esri_mosaic.tif"
            try:
                fetch_mosaic_to_geotiff(bbox, out_path=tif_path, gsd_m=gsd_m)

                patch_count = 0
                for patch_meta in cut_patches_from_raster(
                    raster_path=tif_path,
                    source_tile_id=str(tile_record.id),
                    patch_size=patch_size,
                    overlap_ratio=overlap_ratio,
                    gsd_m=gsd_m,
                    aoi_bbox=bbox if clip_to_bbox else None,
                ):
                    if max_patches is not None and patch_count >= max_patches:
                        logger.info("ingest_patch_limit_reached", max_patches=max_patches)
                        break
                    repo.create_patch(
                        source_tile_id=tile_record.id,
                        center_lon=patch_meta.center_lon,
                        center_lat=patch_meta.center_lat,
                        bbox=patch_meta.bbox,
                        s3_path=patch_meta.s3_key,
                        patch_size=patch_meta.patch_size,
                        gsd_m=patch_meta.gsd_m,
                    )
                    patch_count += 1
                    if patch_count % 100 == 0:
                        session.commit()
                        self.update_state(
                            state="PROGRESS",
                            meta={**stats, "patches_created": patch_count},
                        )

                session.commit()
                repo.mark_tile_processed(tile_record.id)
                session.commit()
                stats["patches_created"] = patch_count
                logger.info("ingest_tile_done", source_id=source_id, patches=patch_count)

            except Exception as exc:
                logger.error("ingest_failed", source_id=source_id, error=str(exc))
                session.rollback()
                raise

    logger.info("ingest_done", **stats)
    return stats
