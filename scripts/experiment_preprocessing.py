#!/usr/bin/env python
"""
Быстрый оффлайн-эксперимент с предобработкой (без пересборки БД/FAISS-индекса).

Идея: у нас уже есть "честный" тест — UAV-снимок и папка с нарезанными
Sentinel-патчами по интересующей области (64x64 / 128x128 px), и известно,
какой патч правильный. Этот скрипт прогоняет SIFT + BFMatcher + RANSAC
НАПРЯМУЮ (минуя BoVW/FAISS coarse-retrieval) для разных вариантов
предобработки и показывает, на каком ранге оказывается правильный патч —
чтобы быстро понять, какая комбинация (понижение разрешения / нормализация
каналов / CLAHE / LCN) реально сокращает разрыв между UAV и Sentinel-2,
до того как тратить время на переобучение словаря и переиндексацию.

Пример:
    python scripts/experiment_preprocessing.py \
        --query uav_crop_64.png \
        --patches-dir ./honest_test/sentinel_128 \
        --truth-file sentinel_patch_00042.png

    # Ограничить перебор конкретными сценариями:
    python scripts/experiment_preprocessing.py \
        --query uav_crop_64.png --patches-dir ./patches --truth-file truth.png \
        --scale 1.0 0.5 0.25 --normalize --lcn
"""
from __future__ import annotations

import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import click
from geoalchemy2.shape import to_shape

from config import configure_logging
from services.features.sift import extract_descriptors
from services.ingestor.storage import download_bytes
from services.db.models import Patch
from services.db.session import SyncSessionLocal
from services.matching.verifier import _ransac_inliers, _ratio_test


@dataclass(frozen=True)
class CandidatePatch:
    name: str
    patch_id: int | None = None
    path: Path | None = None
    s3_path: str | None = None
    center_lat: float | None = None
    center_lon: float | None = None


def _load_patch_files(patches_dir: Path) -> list[CandidatePatch]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return [
        CandidatePatch(name=p.name, path=p)
        for p in sorted(patches_dir.iterdir())
        if p.suffix.lower() in exts
    ]


def _load_db_patches(
    patch_size: int | None,
    product_contains: str | None,
    bbox: tuple[float, float, float, float] | None,
    limit: int | None,
) -> list[CandidatePatch]:
    with SyncSessionLocal() as session:
        rows = session.query(Patch).all()
        if product_contains is not None:
            rows = [p for p in rows if p.source_tile and product_contains in p.source_tile.product_id]
        if patch_size is not None:
            rows = [p for p in rows if p.patch_size == patch_size]

        candidates = []
        for p in rows:
            center = to_shape(p.center)
            if bbox is not None:
                lon_min, lat_min, lon_max, lat_max = bbox
                if not (lon_min <= center.x <= lon_max and lat_min <= center.y <= lat_max):
                    continue
            candidates.append(
                CandidatePatch(
                    name=f"patch_id={p.id}",
                    patch_id=p.id,
                    s3_path=p.s3_path,
                    center_lat=center.y,
                    center_lon=center.x,
                )
            )

    candidates.sort(key=lambda c: c.patch_id or 0)
    return candidates[:limit] if limit is not None else candidates


def _load_candidate_source(candidate: CandidatePatch) -> Path | bytes:
    if candidate.path is not None:
        return candidate.path
    if candidate.s3_path is not None:
        return download_bytes(candidate.s3_path)
    raise ValueError(f"Candidate {candidate.name} has neither path nor s3_path")


def _match_one(
    q_kp,
    q_desc,
    c_kp,
    c_desc,
    lowe_ratio: float,
    ransac_threshold: float,
    min_good_matches: int,
) -> tuple[int, float]:
    """Вернуть (inlier_count, inlier_ratio) между query и одним кандидатом."""
    if q_desc is None or c_desc is None or len(c_desc) < min_good_matches:
        return 0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(q_desc, c_desc, k=2)
    good = _ratio_test(matches, lowe_ratio)
    inliers = _ransac_inliers(q_kp, c_kp, good, ransac_threshold, min_good_matches=min_good_matches)
    ratio = inliers / len(good) if good else 0.0
    return inliers, ratio


def run_config(
    query_path: Path,
    candidates: list[CandidatePatch],
    truth_name: str | None,
    truth_patch_id: int | None,
    query_scale: float,
    patch_scale: float,
    normalize: bool,
    use_clahe: bool,
    use_lcn: bool,
    lowe_ratio: float,
    ransac_threshold: float,
    min_good_matches: int,
) -> dict:
    t0 = time.time()

    kwargs = dict(
        normalize=normalize,
        use_clahe=use_clahe,
        use_lcn=use_lcn,
    )

    q_kp, q_desc = extract_descriptors(query_path, resize_scale=query_scale, **kwargs)
    if q_desc is None:
        return {"error": "no query descriptors", "n_query_kp": 0}

    results = []
    for cand in candidates:
        c_kp, c_desc = extract_descriptors(_load_candidate_source(cand), resize_scale=patch_scale, **kwargs)
        inliers, ratio = _match_one(q_kp, q_desc, c_kp, c_desc, lowe_ratio, ransac_threshold, min_good_matches)
        score = inliers * ratio
        results.append((cand, inliers, ratio, score, len(c_kp) if c_kp else 0))

    # score = inliers * inlier_ratio — см. verifier.py: убирает смещение в
    # пользу "богатых" текстурой тайлов, у которых просто больше keypoints.
    results.sort(key=lambda x: x[3], reverse=True)

    def is_truth(candidate: CandidatePatch) -> bool:
        if truth_patch_id is not None:
            return candidate.patch_id == truth_patch_id
        return truth_name is not None and candidate.name == truth_name

    rank = next((i + 1 for i, (cand, *_rest) in enumerate(results) if is_truth(cand)), None)
    truth_inliers = next((inl for cand, inl, _r, _s, _n in results if is_truth(cand)), 0)
    truth_ratio = next((r for cand, _i, r, _s, _n in results if is_truth(cand)), 0.0)

    return {
        "n_query_kp": len(q_kp),
        "rank": rank,
        "n_candidates": len(results),
        "truth_inliers": truth_inliers,
        "truth_ratio": round(truth_ratio, 3),
        "top5": [
            {
                "name": cand.name,
                "patch_id": cand.patch_id,
                "inliers": inliers,
                "ratio": round(ratio, 3),
                "score": round(score, 2),
                "n_kp": n_kp,
                "center_lat": cand.center_lat,
                "center_lon": cand.center_lon,
            }
            for cand, inliers, ratio, score, n_kp in results[:5]
        ],
        "elapsed_s": round(time.time() - t0, 1),
    }


@click.command()
@click.option("--query", required=True, type=click.Path(exists=True), help="UAV query изображение")
@click.option("--patches-dir", type=click.Path(exists=True, file_okay=False), help="Папка с Sentinel-патчами")
@click.option("--from-db", is_flag=True, default=False, help="Брать Sentinel-патчи из PostgreSQL/MinIO")
@click.option("--truth-file", default=None, help="Имя файла правильного патча внутри --patches-dir")
@click.option("--truth-patch-id", default=None, type=int, help="patch_id правильного патча при --from-db")
@click.option("--patch-size", default=None, type=int, help="Фильтр patch_size при --from-db")
@click.option("--product-contains", default=None, help="Фильтр source_tiles.product_id при --from-db")
@click.option("--bbox", default=None, help="Фильтр центров патчей: lon_min,lat_min,lon_max,lat_max")
@click.option("--limit", default=None, type=int, help="Ограничить число кандидатов после фильтров")
@click.option("--scale", "scales", multiple=True, type=float, default=(1.0, 0.5, 0.25, 0.1), show_default=True,
              help="Обратная совместимость: query и patch scale, если не заданы отдельные scale")
@click.option("--query-scale", "query_scales", multiple=True, type=float,
              help="Значения resize_scale только для UAV query")
@click.option("--patch-scale", "patch_scales", multiple=True, type=float,
              help="Значения resize_scale только для Sentinel-патчей")
@click.option("--normalize/--no-normalize-sweep", "sweep_normalize", default=True,
              help="Перебирать normalize=True/False (по умолчанию перебор)")
@click.option("--clahe/--no-clahe-sweep", "sweep_clahe", default=False,
              help="Добавить в перебор use_clahe=True")
@click.option("--lcn/--no-lcn-sweep", "sweep_lcn", default=False,
              help="Добавить в перебор use_lcn=True (нормализованная карта)")
@click.option("--lowe-ratio", default=0.75, show_default=True)
@click.option("--ransac-threshold", default=5.0, show_default=True)
@click.option("--min-good-matches", default=4, show_default=True)
@click.option("--log-level", default="WARNING", show_default=True)
def main(
    query, patches_dir, from_db, truth_file, truth_patch_id,
    patch_size, product_contains, bbox, limit,
    scales, query_scales, patch_scales, sweep_normalize, sweep_clahe, sweep_lcn,
    lowe_ratio, ransac_threshold, min_good_matches, log_level,
):
    configure_logging(log_level)

    query_path = Path(query)
    bbox_tuple = None
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise click.ClickException("--bbox must contain 4 comma-separated values")
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    if from_db:
        if truth_patch_id is None:
            raise click.ClickException("--truth-patch-id обязателен при --from-db")
        candidates = _load_db_patches(patch_size, product_contains, bbox_tuple, limit)
    else:
        if patches_dir is None or truth_file is None:
            raise click.ClickException("--patches-dir и --truth-file обязательны без --from-db")
        candidates = _load_patch_files(Path(patches_dir))
        if not any(p.name == truth_file for p in candidates):
            raise click.ClickException(f"Файл {truth_file} не найден в {patches_dir}")

    if not candidates:
        raise click.ClickException("Нет патчей-кандидатов после фильтров")
    if from_db and not any(p.patch_id == truth_patch_id for p in candidates):
        raise click.ClickException(f"patch_id={truth_patch_id} не найден среди кандидатов")

    query_scales = query_scales or scales
    patch_scales = patch_scales or scales

    click.echo("=" * 78)
    click.echo("  GeoVision — эксперимент с предобработкой (UAV vs Sentinel, честный тест)")
    click.echo("=" * 78)
    click.echo(f"Query: {query_path.name}")
    truth_label = f"patch_id={truth_patch_id}" if truth_patch_id is not None else truth_file
    click.echo(f"Патчей в базе: {len(candidates)}  |  Truth: {truth_label}")
    click.echo("")

    normalize_opts = [False, True] if sweep_normalize else [False]
    clahe_opts = [False, True] if sweep_clahe else [False]
    lcn_opts = [False, True] if sweep_lcn else [False]

    header = (
        f"{'q_s':>6} {'p_s':>6} {'norm':>5} {'clahe':>6} {'lcn':>5} "
        f"{'q_kp':>6} {'rank':>6} {'inliers':>8} {'ratio':>6} {'time_s':>7}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    all_results = []
    for q_scale, p_scale, norm, clahe, lcn in itertools.product(query_scales, patch_scales, normalize_opts, clahe_opts, lcn_opts):
        res = run_config(
            query_path, candidates, truth_file, truth_patch_id,
            query_scale=q_scale, patch_scale=p_scale, normalize=norm, use_clahe=clahe, use_lcn=lcn,
            lowe_ratio=lowe_ratio, ransac_threshold=ransac_threshold, min_good_matches=min_good_matches,
        )
        if "error" in res:
            click.echo(f"{q_scale:>6} {p_scale:>6} {norm!s:>5} {clahe!s:>6} {lcn!s:>5}   -- {res['error']} --")
            continue

        rank_str = str(res["rank"]) if res["rank"] else "NOT FOUND"
        line = (
            f"{q_scale:>6} {p_scale:>6} {norm!s:>5} {clahe!s:>6} {lcn!s:>5} "
            f"{res['n_query_kp']:>6} {rank_str:>6} {res['truth_inliers']:>8} "
            f"{res['truth_ratio']:>6} {res['elapsed_s']:>7}"
        )
        click.echo(line)
        all_results.append({"query_scale": q_scale, "patch_scale": p_scale, "normalize": norm, "clahe": clahe, "lcn": lcn, **res})

    click.echo("")
    valid = [r for r in all_results if r.get("rank")]
    if valid:
        best = min(valid, key=lambda r: r["rank"])
        click.secho(
            f"Лучшая конфигурация: query_scale={best['query_scale']} patch_scale={best['patch_scale']} "
            f"normalize={best['normalize']} "
            f"clahe={best['clahe']} lcn={best['lcn']} → rank={best['rank']} "
            f"(inliers={best['truth_inliers']}, было бы top-{best['n_candidates']})",
            fg="green",
        )
    else:
        click.secho("Ни в одной конфигурации правильный патч не найден среди кандидатов.", fg="red")


if __name__ == "__main__":
    main()
