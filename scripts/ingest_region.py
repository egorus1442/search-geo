#!/usr/bin/env python
"""
CLI скрипт: запустить ingestion синхронно (без Celery) для отладки и первичного запуска.

Источник — Esri World Imagery (заменил Sentinel-2/CDSE).

Пример (эталон ~1 м вокруг Багаряка, патчи 640 px = 640 м footprint):
    python scripts/ingest_region.py \
        --bbox 61.47 56.17 61.57 56.24 \
        --gsd 1.0 \
        --patch-size 640
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
@click.option("--gsd", "gsd_m", default=None, type=float, help="Целевой GSD эталона, м/пкс (по умолчанию из settings)")
@click.option("--patch-size", default=None, type=int, help="Размер патча в пикселях (footprint = patch_size * gsd)")
@click.option("--overlap-ratio", default=None, type=float, help="Перекрытие патчей (доля)")
@click.option("--max-patches", default=None, type=int, help="Лимит числа патчей (отладка)")
@click.option("--clip-to-bbox", is_flag=True, default=False, help="Оставлять только патчи, пересекающие bbox")
@click.option("--run-label", default=None, help="Метка для повторной нарезки того же bbox")
@click.option("--log-level", default="INFO", show_default=True)
def main(
    bbox,
    gsd_m,
    patch_size,
    overlap_ratio,
    max_patches,
    clip_to_bbox,
    run_label,
    log_level,
):
    """Скачать мозаику Esri World Imagery и нарезать на патчи."""
    configure_logging(log_level)
    ensure_bucket()

    bbox_list = list(bbox)
    logger.info("ingest_start", bbox=bbox_list, gsd_m=gsd_m, patch_size=patch_size)

    # Запускаем синхронно через .apply() — без брокера
    result = run_ingest.apply(
        kwargs={
            "bbox": bbox_list,
            "gsd_m": gsd_m,
            "patch_size": patch_size,
            "overlap_ratio": overlap_ratio,
            "max_patches": max_patches,
            "clip_to_bbox": clip_to_bbox,
            "run_label": run_label,
        }
    ).get()

    click.echo(f"\nDone: {result}")


if __name__ == "__main__":
    main()
