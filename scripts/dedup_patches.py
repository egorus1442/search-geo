#!/usr/bin/env python
"""
CLI скрипт: удалить патчи конкретного (дублирующего) source_tile_id из БД + MinIO.

По умолчанию делает dry-run (только считает и печатает, что было бы удалено).
Передайте --execute чтобы реально удалить.

Пример:
    python scripts/dedup_patches.py --source-tile-id 3dbfb4c8-c383-4fb9-84d7-d8ee605f3bec
    python scripts/dedup_patches.py --source-tile-id 3dbfb4c8-c383-4fb9-84d7-d8ee605f3bec --execute
    python scripts/dedup_patches.py --source-tile-id 3dbfb4c8-c383-4fb9-84d7-d8ee605f3bec --execute --drop-source-tile
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from minio.error import S3Error
from sqlalchemy import delete, select

from config import configure_logging, get_logger, get_settings
from services.db.models import Patch, SourceTile
from services.db.session import SyncSessionLocal
from services.ingestor.storage import _get_client

logger = get_logger(__name__)


@click.command()
@click.option("--source-tile-id", required=True, help="UUID source_tile, чьи патчи нужно удалить")
@click.option("--execute", is_flag=True, default=False, help="Реально удалить (иначе только dry-run)")
@click.option("--drop-source-tile", is_flag=True, default=False, help="Также удалить строку из source_tiles")
@click.option("--log-level", default="INFO", show_default=True)
def main(source_tile_id, execute, drop_source_tile, log_level):
    configure_logging(log_level)
    settings = get_settings()
    bucket = settings.minio_bucket

    with SyncSessionLocal() as session:
        rows = session.execute(
            select(Patch.id, Patch.s3_path).where(Patch.source_tile_id == source_tile_id)
        ).all()

    n = len(rows)
    click.echo(f"Найдено {n} патчей с source_tile_id={source_tile_id}")
    if n == 0:
        click.echo("Нечего удалять.")
        return

    if not execute:
        click.echo("Dry-run: ничего не удалено. Передайте --execute для реального удаления.")
        click.echo(f"Пример patch_id для удаления: {[r[0] for r in rows[:5]]}")
        return

    client = _get_client()
    removed_objects = 0
    missing_objects = 0
    for patch_id, s3_path in rows:
        try:
            client.remove_object(bucket, s3_path)
            removed_objects += 1
        except S3Error as exc:
            missing_objects += 1
            logger.warning("minio_remove_error", patch_id=patch_id, s3_path=s3_path, error=str(exc))

    click.echo(f"MinIO: удалено {removed_objects} объектов, {missing_objects} отсутствовали/ошиблись.")

    with SyncSessionLocal() as session:
        result = session.execute(
            delete(Patch).where(Patch.source_tile_id == source_tile_id)
        )
        session.commit()
        click.echo(f"БД: удалено {result.rowcount} строк из patches.")

        if drop_source_tile:
            session.execute(delete(SourceTile).where(SourceTile.id == source_tile_id))
            session.commit()
            click.echo(f"БД: удалена строка source_tiles id={source_tile_id}.")

    click.echo("\nГотово. Теперь пересоберите словарь+индекс:")
    click.echo("  python scripts/build_vocab.py")


if __name__ == "__main__":
    main()
