#!/usr/bin/env python
"""
Self-consistency тест: берёт реальный патч из базы, обрезает и немного
поворачивает его (имитация фото с дрона под другим углом/масштабом),
отправляет в /api/v1/localize и проверяет, находит ли сервис исходный патч.

Запускать ИЗНУТРИ контейнера api (там уже настроены хосты postgres/minio):

    docker-compose exec api python scripts/test_selfcheck.py
    docker-compose exec api python scripts/test_selfcheck.py --patch-id 42
    docker-compose exec api python scripts/test_selfcheck.py --crop-ratio 0.6 --rotate-deg 15
    docker-compose exec api python scripts/test_selfcheck.py --out /data/debug_query.png

Что делает:
  1. Берёт патч (случайный либо по --patch-id) из Postgres + скачивает PNG из MinIO
  2. Вырезает центральный кроп (crop-ratio от размера патча)
  3. Поворачивает на rotate-deg градусов (с расширением холста, без обрезки контента)
  4. Отправляет получившееся изображение в POST /api/v1/localize
  5. Смотрит, есть ли исходный patch_id среди top-N кандидатов, на каком ранге,
     с каким inlier_count/confidence, и насколько предсказанные координаты
     совпадают с истинным центром патча
"""
import io
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
import requests
from PIL import Image

from services.db.session import SyncSessionLocal
from services.index.metadata_store import PatchRepo
from services.ingestor.storage import download_bytes


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def make_query_image(png_bytes: bytes, crop_ratio: float, rotate_deg: float) -> bytes:
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    w, h = img.size

    # Центральный кроп
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))

    # Поворот с расширением холста (не обрезаем контент, углы станут чёрными —
    # это не мешает SIFT, т.к. на однородном чёрном фоне ключевых точек нет)
    rotated = cropped.rotate(rotate_deg, resample=Image.BICUBIC, expand=True, fillcolor=(0, 0, 0))

    buf = io.BytesIO()
    rotated.save(buf, format="PNG")
    return buf.getvalue()


def pick_patch(patch_id: int | None) -> dict:
    with SyncSessionLocal() as session:
        repo = PatchRepo(session)
        if patch_id is not None:
            found = repo.get_patches_by_ids([patch_id])
            if not found:
                raise click.ClickException(f"Patch {patch_id} not found in DB")
            return found[0]

        all_ids = repo.get_all_patch_ids()
        if not all_ids:
            raise click.ClickException(
                "В базе нет патчей. Сначала запусти ingestion (POST /api/v1/admin/ingest)."
            )
        chosen = random.choice(all_ids)
        return repo.get_patches_by_ids([chosen])[0]


@click.command()
@click.option("--patch-id", default=None, type=int, help="Конкретный patch_id (иначе случайный)")
@click.option("--crop-ratio", default=0.7, show_default=True, help="Доля от размера патча (0-1)")
@click.option("--rotate-deg", default=12.0, show_default=True, help="Угол поворота в градусах")
@click.option("--top-n", default=10, show_default=True)
@click.option("--api-url", default="http://localhost:8000", show_default=True)
@click.option("--out", default=None, type=click.Path(), help="Сохранить получившийся query-файл для просмотра")
def main(patch_id, crop_ratio, rotate_deg, top_n, api_url, out):
    click.echo("=" * 72)
    click.echo("  GeoVision — Self-consistency check (crop + rotate)")
    click.echo("=" * 72)

    patch = pick_patch(patch_id)
    click.echo(f"\n[1] Исходный патч: id={patch['patch_id']}")
    click.echo(f"    center: lat={patch['center_lat']:.6f} lon={patch['center_lon']:.6f}")
    click.echo(f"    s3_path: {patch['s3_path']}")

    click.echo(f"\n[2] Скачиваю оригинал из MinIO...")
    original_bytes = download_bytes(patch["s3_path"])

    click.echo(f"[3] Делаю crop={crop_ratio} + rotate={rotate_deg}°...")
    query_bytes = make_query_image(original_bytes, crop_ratio, rotate_deg)

    if out:
        Path(out).write_bytes(query_bytes)
        click.echo(f"    Сохранено: {out}")

    click.echo(f"\n[4] Отправляю в {api_url}/api/v1/localize (top_n={top_n})...")
    resp = requests.post(
        f"{api_url}/api/v1/localize",
        files={"image": ("query.png", query_bytes, "image/png")},
        data={"top_n": top_n},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()

    click.echo(f"\n    Status: {data['status']}, time: {data.get('processing_time_ms', '?')} ms")
    click.echo(f"    Candidates returned: {len(data['candidates'])}")

    click.echo("\n" + "-" * 72)
    click.echo(f"{'#':<3} {'patch_id':>10} {'inliers':>8} {'conf':>6}  {'dist_km':>8}")
    click.echo("-" * 72)

    found_rank = None
    for c in data["candidates"]:
        dist = _haversine_km(patch["center_lat"], patch["center_lon"], c["center_lat"], c["center_lon"])
        marker = "  <== ORIGINAL" if c["patch_id"] == patch["patch_id"] else ""
        click.echo(
            f"{c['rank']:<3} {c['patch_id']:>10} {c['inlier_count']:>8} "
            f"{c['confidence']:>6.2f}  {dist:>8.3f}{marker}"
        )
        if c["patch_id"] == patch["patch_id"]:
            found_rank = c["rank"]

    click.echo("-" * 72)

    if found_rank == 1:
        click.secho(f"\n✅ PASS: исходный патч найден на 1-м месте.", fg="green")
    elif found_rank is not None:
        click.secho(f"\n⚠️  PARTIAL: исходный патч найден, но на ранге {found_rank} (не 1-м).", fg="yellow")
    else:
        click.secho(
            "\n❌ FAIL: исходный патч НЕ найден среди top-N. "
            "Попробуй увеличить crop-ratio, уменьшить rotate-deg, "
            "или проверить другой patch-id (текстура могла быть слабой).",
            fg="red",
        )


if __name__ == "__main__":
    main()
