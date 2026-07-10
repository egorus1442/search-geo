#!/usr/bin/env python
"""
Метрический harness recall@K для coarse-стадии (BoVW vs VLAD vs ...).

Зачем
-----
Раньше выводы о качестве coarse-фильтра были анекдотичными («правильный патч
оказался на 27-м месте»). Этот скрипт даёт объективную цифру: на каком ранге
coarse-стадия ставит ИСТИННЫЙ патч и попадает ли он в top-K — то самое, что
определяет, доживёт ли верный кандидат до дорогой RANSAC-верификации.

recall@K здесь = «истинный патч попал в top-K coarse-кандидатов» (1/0 для
одного query). Прогоняя несколько методов на одной выборке патчей, можно
честно сравнить BoVW и VLAD ДО пересборки боевого индекса.

Важно: coarse-энкодеры обучаются на выбранной подвыборке патчей (--fit),
чтобы мерить именно retrieval, не завися от того, что лежит в /data/index.
Verifier (SIFT+RANSAC) тут НЕ участвует — измеряем только coarse-recall.

Примеры
-------
    # Село Багаряк (структурный контент), сравнить оба метода:
    python scripts/eval_recall.py \
        --query geovision/bagaryak_query.jpg \
        --truth-patch-id 1488 \
        --product-contains T41 \
        --methods bovw,vlad --fit

    # Лес Харьков (низкотекстурный, ожидаемый провал классики):
    python scripts/eval_recall.py \
        --query external_uav_kharkiv_botanic_query_wide_570m.jpg \
        --center-lon 36.2402455 --center-lat 50.028757 --truth-tol-km 0.4 \
        --methods bovw,vlad --fit
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
from geoalchemy2.shape import to_shape

from config import configure_logging
from services.db.models import Patch
from services.db.session import SyncSessionLocal
from services.features.sift import extract_patch_descriptors, extract_query_descriptors
from services.features.vlad import VladEncoder
from services.features.vocabulary import Vocabulary
from services.ingestor.storage import download_bytes


def _haversine_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


IMAGE_METHODS = {"dino", "dino_vlad"}


def _build_encoder(method: str, desc_list: list[np.ndarray], image_list: list[bytes]):
    """
    Обучить/подготовить временный coarse-энкодер на выбранных патчах.

    vlad/bovw — на SIFT-дескрипторах (desc_list); dino/dino_vlad — на картинках
    (image_list): dino не обучается (pretrained), dino_vlad учит k-means по
    DINOv2-токенам.
    """
    if method == "vlad":
        enc = VladEncoder()
        enc.fit(lambda: iter(desc_list))
        return enc
    if method == "bovw":
        enc = Vocabulary()
        enc.fit(iter(desc_list))
        return enc
    if method == "dino":
        from services.features.dino import DinoEncoder  # ленивый импорт torch/timm
        return DinoEncoder()
    if method == "dino_vlad":
        from services.features.dino import DinoVladEncoder
        enc = DinoVladEncoder()
        enc.fit_images(lambda: iter(image_list))
        return enc
    raise click.ClickException(f"Unknown method: {method}")


@click.command()
@click.option("--query", required=True, type=click.Path(exists=True), help="UAV query изображение")
@click.option("--truth-patch-id", default=None, type=int, help="patch_id истинного патча")
@click.option("--center-lon", default=None, type=float, help="Долгота истинной точки (если нет truth-patch-id)")
@click.option("--center-lat", default=None, type=float, help="Широта истинной точки")
@click.option("--truth-tol-km", default=0.4, show_default=True,
              help="Радиус: ближайший патч в этом радиусе от center считается истинным")
@click.option("--product-contains", default=None, help="Фильтр source_tiles.product_id")
@click.option("--patch-size", default=None, type=int, help="Фильтр patch_size")
@click.option("--bbox", default=None, help="Фильтр центров: lon_min,lat_min,lon_max,lat_max")
@click.option("--methods", default="bovw,vlad", show_default=True,
              help="Список методов через запятую: bovw,vlad,dino,dino_vlad")
@click.option("--ks", default="1,5,10,50,100", show_default=True, help="Значения K для recall@K")
@click.option("--fit/--no-fit", default=True, show_default=True,
              help="Обучать временный энкодер на выбранных патчах (иначе загрузить боевой)")
@click.option("--log-level", default="WARNING", show_default=True)
def main(
    query, truth_patch_id, center_lon, center_lat, truth_tol_km,
    product_contains, patch_size, bbox, methods, ks, fit, log_level,
):
    configure_logging(log_level)

    if truth_patch_id is None and (center_lon is None or center_lat is None):
        raise click.ClickException("Задай либо --truth-patch-id, либо --center-lon/--center-lat")

    ks_list = sorted({int(x) for x in ks.split(",") if x.strip()})
    method_list = [m.strip().lower() for m in methods.split(",") if m.strip()]

    bbox_tuple = None
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise click.ClickException("--bbox must contain 4 comma-separated values")
        bbox_tuple = tuple(parts)

    # ── Загрузка кандидатов из БД ────────────────────────────────────────────────
    with SyncSessionLocal() as session:
        rows = session.query(Patch).all()
        if product_contains is not None:
            rows = [p for p in rows if p.source_tile and product_contains in p.source_tile.product_id]
        if patch_size is not None:
            rows = [p for p in rows if p.patch_size == patch_size]

        candidates = []
        for p in rows:
            center = to_shape(p.center)
            if bbox_tuple is not None:
                lo_lon, lo_lat, hi_lon, hi_lat = bbox_tuple
                if not (lo_lon <= center.x <= hi_lon and lo_lat <= center.y <= hi_lat):
                    continue
            candidates.append({
                "patch_id": p.id,
                "center_lon": center.x,
                "center_lat": center.y,
                "s3_path": p.s3_path,
            })

    if not candidates:
        raise click.ClickException("Нет патчей-кандидатов после фильтров")

    # ── Определить истинный patch_id ─────────────────────────────────────────────
    if truth_patch_id is None:
        nearest = min(
            candidates,
            key=lambda c: _haversine_km(center_lon, center_lat, c["center_lon"], c["center_lat"]),
        )
        dist_km = _haversine_km(center_lon, center_lat, nearest["center_lon"], nearest["center_lat"])
        if dist_km > truth_tol_km:
            raise click.ClickException(
                f"Ближайший патч на {dist_km:.2f} км > tol={truth_tol_km} км — истинного нет в выборке"
            )
        truth_patch_id = nearest["patch_id"]
        click.echo(f"truth auto-detected: patch_id={truth_patch_id} ({dist_km*1000:.0f} м от точки)")

    if not any(c["patch_id"] == truth_patch_id for c in candidates):
        raise click.ClickException(f"patch_id={truth_patch_id} не найден среди кандидатов")

    # ── Дескрипторы query + всех кандидатов (кэш) ────────────────────────────────
    # ВАЖНО: query и патчи должны идти через ту же предобработку, что и боевой
    # путь (localize): query — с PREPROCESS_QUERY_RESIZE_SCALE (согласование GSD
    # UAV↔Sentinel), патчи — с PREPROCESS_PATCH_RESIZE_SCALE. Иначе SIFT-дескрипторы
    # живут в разных масштабах и не сопоставляются (см. историю: query 0.12 м/пкс
    # vs патч 10 м/пкс → scale ≈ 0.06).
    query_bytes = Path(query).read_bytes()
    _, query_desc = extract_query_descriptors(query_bytes)
    if query_desc is None:
        raise click.ClickException("No SIFT descriptors extracted from query")

    click.echo(f"query={Path(query).name} q_features={len(query_desc)} candidates={len(candidates)}")
    click.echo("loading candidate descriptors...")
    desc_by_id: dict[int, np.ndarray] = {}
    bytes_by_id: dict[int, bytes] = {}
    for c in candidates:
        try:
            raw = download_bytes(c["s3_path"])
            bytes_by_id[c["patch_id"]] = raw
            _, d = extract_patch_descriptors(raw)
            if d is not None:
                desc_by_id[c["patch_id"]] = d
        except Exception as exc:
            click.echo(f"warn patch_id={c['patch_id']} error={exc}", err=True)

    valid = [c for c in candidates if c["patch_id"] in desc_by_id]
    desc_list = [desc_by_id[c["patch_id"]] for c in valid]
    image_list = [bytes_by_id[c["patch_id"]] for c in valid]
    click.echo(f"candidates with descriptors: {len(valid)}")

    if not fit:
        raise click.ClickException("--no-fit пока не поддержан в harness (нужен боевой энкодер под метод)")

    # ── Прогон методов ───────────────────────────────────────────────────────────
    click.echo("")
    header = f"{'method':>8} {'dim':>6} {'rank':>6} " + " ".join(f"R@{k:<4}" for k in ks_list) + f" {'fit_s':>7} {'enc_s':>7}"
    click.echo(header)
    click.echo("-" * len(header))

    for method in method_list:
        is_image = method in IMAGE_METHODS
        t0 = time.time()
        enc = _build_encoder(method, desc_list, image_list)
        t_fit = time.time() - t0

        t1 = time.time()
        if is_image:
            # dino/dino_vlad: кодируем из КАРТИНКИ (оригинал query/патча, без
            # SIFT-препроцессинга) — DINOv2 сам ресайзит до GLOBAL_IMAGE_SIZE.
            q_vec = enc.encode_image(query_bytes)
            dists = []
            for c in valid:
                v = enc.encode_image(bytes_by_id[c["patch_id"]])
                dists.append((float(np.linalg.norm(q_vec - v)), c["patch_id"]))
        else:
            q_vec = enc.encode(query_desc)
            dists = []
            for c in valid:
                v = enc.encode(desc_by_id[c["patch_id"]])
                dists.append((float(np.linalg.norm(q_vec - v)), c["patch_id"]))
        t_enc = time.time() - t1

        dists.sort(key=lambda x: x[0])
        rank = next((i + 1 for i, (_, pid) in enumerate(dists) if pid == truth_patch_id), None)
        recalls = ["1" if (rank is not None and rank <= k) else "0" for k in ks_list]

        rank_str = str(rank) if rank else "MISS"
        click.echo(
            f"{method:>8} {enc.dim:>6} {rank_str:>6} "
            + " ".join(f"{r:<5}" for r in recalls)
            + f" {t_fit:>7.1f} {t_enc:>7.1f}"
        )

    click.echo("")
    click.echo("recall@K=1 → истинный патч дожил бы до RANSAC (top_n_coarse=K). Чем меньше rank — тем лучше.")


if __name__ == "__main__":
    main()
