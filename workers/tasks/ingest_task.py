"""
Celery task: скачать Sentinel-2 снимки по bbox и нарезать на патчи.

Очередь: ingest
"""
import shutil
import tempfile
import uuid
from datetime import date
from pathlib import Path

from workers.celery_app import app
from config import get_logger
from services.db.session import SyncSessionLocal
from services.index.metadata_store import PatchRepo
from services.ingestor.cdse_client import CDSEClient
from services.ingestor.tile_cutter import cut_patches, extract_safe
from services.ingestor.storage import ensure_bucket

logger = get_logger(__name__)


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
    date_from: str,
    date_to: str,
    cloud_cover_max: float = 20.0,
    task_db_id: str | None = None,
    max_products: int | None = None,
    max_patches_per_product: int | None = None,
    patch_size: int | None = None,
    clip_to_bbox: bool = False,
    run_label: str | None = None,
) -> dict:
    """
    Скачать продукты Sentinel-2 по параметрам и нарезать на патчи.

    bbox: [lon_min, lat_min, lon_max, lat_max]
    date_from / date_to: "YYYY-MM-DD"
    """
    logger.info("ingest_start", bbox=bbox, date_from=date_from, date_to=date_to)

    ensure_bucket()

    d_from = date.fromisoformat(date_from)
    d_to = date.fromisoformat(date_to)

    client = CDSEClient()
    products = client.search(
        bbox=bbox,
        date_from=d_from,
        date_to=d_to,
        cloud_cover_max=cloud_cover_max,
        max_results=max_products or 100,
    )
    logger.info("ingest_found_products", count=len(products))

    stats = {"products_found": len(products), "tiles_downloaded": 0, "patches_created": 0, "skipped": 0}

    with SyncSessionLocal() as session:
        repo = PatchRepo(session)

        for product in products:
            product_id: str = product["Id"]
            product_name: str = product.get("Name", product_id)
            source_product_id = f"{product_id}:{run_label}" if run_label else product_id

            # Пропустить уже обработанные
            if repo.tile_exists(source_product_id):
                logger.info("ingest_skip_existing", product_id=source_product_id)
                stats["skipped"] += 1
                continue

            # Создаём запись SourceTile
            content_date = product.get("ContentDate", {}).get("Start", date_from)
            cloud_cover = None
            for attr in product.get("Attributes", []):
                if attr.get("Name") == "cloudCover":
                    cloud_cover = attr.get("Value")

            tile_record = repo.create_source_tile(
                product_id=source_product_id,
                bbox=bbox,  # упрощение: используем запросный bbox
                date_acq=content_date,
                cloud_cover=cloud_cover,
            )
            session.commit()

            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                try:
                    zip_path = client.download(product_id, output_dir=tmp_path)
                    safe_dir = extract_safe(zip_path, tmp_path / "safe")
                    stats["tiles_downloaded"] += 1

                    patch_count = 0
                    for patch_meta in cut_patches(
                        safe_dir=safe_dir,
                        source_tile_id=str(tile_record.id),
                        patch_size=patch_size,
                        aoi_bbox=bbox if clip_to_bbox else None,
                    ):
                        if max_patches_per_product is not None and patch_count >= max_patches_per_product:
                            logger.info(
                                "ingest_patch_limit_reached",
                                product_id=product_id,
                                max_patches=max_patches_per_product,
                            )
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
                                meta={**stats, "patches_created": stats["patches_created"] + patch_count},
                            )

                    session.commit()
                    repo.mark_tile_processed(tile_record.id)
                    session.commit()
                    stats["patches_created"] += patch_count
                    logger.info("ingest_tile_done", product_id=product_id, patches=patch_count)

                except Exception as exc:
                    logger.error("ingest_tile_failed", product_id=product_id, error=str(exc))
                    session.rollback()
                    # Не прерываем весь джоб из-за одного тайла

    logger.info("ingest_done", **stats)
    return stats
