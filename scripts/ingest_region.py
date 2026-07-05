#!/usr/bin/env python
"""
CLI скрипт: запустить ingestion синхронно (без Celery) для отладки и первичного запуска.

Пример:
    python scripts/ingest_region.py \
        --bbox 35.0 50.0 40.0 55.0 \
        --date-from 2023-06-01 \
        --date-to 2024-09-30 \
        --cloud-cover 20
"""
import sys
from pathlib import Path

# Добавить корень проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from config import configure_logging, get_logger
from services.ingestor.storage import ensure_bucket
from workers.tasks.ingest_task import run_ingest

logger = get_logger(__name__)


@click.command()
@click.option("--bbox", nargs=4, type=float, required=True, metavar="LON_MIN LAT_MIN LON_MAX LAT_MAX")
@click.option("--date-from", required=True, help="YYYY-MM-DD")
@click.option("--date-to", required=True, help="YYYY-MM-DD")
@click.option("--cloud-cover", default=20.0, type=float, show_default=True)
@click.option("--max-products", default=None, type=int, help="Limit CDSE products to download")
@click.option("--max-patches-per-product", default=None, type=int, help="Limit patches cut from each product")
@click.option("--patch-size", default=None, type=int, help="Override patch size in pixels")
@click.option("--clip-to-bbox", is_flag=True, default=False, help="Only keep patches intersecting the requested bbox")
@click.option("--run-label", default=None, help="Optional label to allow reprocessing the same CDSE product")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    bbox,
    date_from,
    date_to,
    cloud_cover,
    max_products,
    max_patches_per_product,
    patch_size,
    clip_to_bbox,
    run_label,
    log_level,
):
    """Скачать Sentinel-2 снимки и нарезать на патчи."""
    configure_logging(log_level)
    ensure_bucket()

    bbox_list = list(bbox)
    logger.info(
        "ingest_start",
        bbox=bbox_list,
        date_from=date_from,
        date_to=date_to,
        cloud_cover=cloud_cover,
    )

    # Запускаем синхронно через .apply() — без брокера
    result = run_ingest.apply(
        kwargs={
            "bbox": bbox_list,
            "date_from": date_from,
            "date_to": date_to,
            "cloud_cover_max": cloud_cover,
            "max_products": max_products,
            "max_patches_per_product": max_patches_per_product,
            "patch_size": patch_size,
            "clip_to_bbox": clip_to_bbox,
            "run_label": run_label,
        }
    ).get()

    click.echo(f"\nDone: {result}")


if __name__ == "__main__":
    main()
