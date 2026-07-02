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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from services.features.sift import extract_descriptors
from services.matching.verifier import _ransac_inliers, _ratio_test
import cv2


def _load_patch_files(patches_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted(p for p in patches_dir.iterdir() if p.suffix.lower() in exts)


def _match_one(
    q_kp,
    q_desc,
    c_kp,
    c_desc,
    lowe_ratio: float,
    ransac_threshold: float,
    min_good_matches: int,
) -> int:
    """Вернуть число RANSAC-инлайеров между query и одним кандидатом."""
    if q_desc is None or c_desc is None or len(c_desc) < min_good_matches:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(q_desc, c_desc, k=2)
    good = _ratio_test(matches, lowe_ratio)
    return _ransac_inliers(q_kp, c_kp, good, ransac_threshold)


def run_config(
    query_path: Path,
    patch_files: list[Path],
    truth_name: str,
    resize_scale: float,
    normalize: bool,
    use_clahe: bool,
    use_lcn: bool,
    lowe_ratio: float,
    ransac_threshold: float,
    min_good_matches: int,
) -> dict:
    t0 = time.time()

    kwargs = dict(
        resize_scale=resize_scale,
        normalize=normalize,
        use_clahe=use_clahe,
        use_lcn=use_lcn,
    )

    q_kp, q_desc = extract_descriptors(query_path, **kwargs)
    if q_desc is None:
        return {"error": "no query descriptors", "n_query_kp": 0}

    results = []
    for pf in patch_files:
        c_kp, c_desc = extract_descriptors(pf, **kwargs)
        inliers = _match_one(q_kp, q_desc, c_kp, c_desc, lowe_ratio, ransac_threshold, min_good_matches)
        results.append((pf.name, inliers, len(c_kp) if c_kp else 0))

    results.sort(key=lambda x: x[1], reverse=True)

    rank = next((i + 1 for i, (name, _, _) in enumerate(results) if name == truth_name), None)
    truth_inliers = next((inl for name, inl, _ in results if name == truth_name), 0)

    return {
        "n_query_kp": len(q_kp),
        "rank": rank,
        "n_candidates": len(results),
        "truth_inliers": truth_inliers,
        "top5": results[:5],
        "elapsed_s": round(time.time() - t0, 1),
    }


@click.command()
@click.option("--query", required=True, type=click.Path(exists=True), help="UAV query изображение")
@click.option("--patches-dir", required=True, type=click.Path(exists=True, file_okay=False), help="Папка с Sentinel-патчами")
@click.option("--truth-file", required=True, help="Имя файла правильного патча внутри --patches-dir")
@click.option("--scale", "scales", multiple=True, type=float, default=(1.0, 0.5, 0.25, 0.1), show_default=True,
              help="Значения resize_scale для перебора (можно указать несколько раз)")
@click.option("--normalize/--no-normalize-sweep", "sweep_normalize", default=True,
              help="Перебирать normalize=True/False (по умолчанию перебор)")
@click.option("--clahe/--no-clahe-sweep", "sweep_clahe", default=False,
              help="Добавить в перебор use_clahe=True")
@click.option("--lcn/--no-lcn-sweep", "sweep_lcn", default=False,
              help="Добавить в перебор use_lcn=True (нормализованная карта)")
@click.option("--lowe-ratio", default=0.75, show_default=True)
@click.option("--ransac-threshold", default=5.0, show_default=True)
@click.option("--min-good-matches", default=4, show_default=True)
def main(
    query, patches_dir, truth_file,
    scales, sweep_normalize, sweep_clahe, sweep_lcn,
    lowe_ratio, ransac_threshold, min_good_matches,
):
    query_path = Path(query)
    patches_dir = Path(patches_dir)
    patch_files = _load_patch_files(patches_dir)

    if not patch_files:
        raise click.ClickException(f"Нет изображений в {patches_dir}")
    if not any(p.name == truth_file for p in patch_files):
        raise click.ClickException(f"Файл {truth_file} не найден в {patches_dir}")

    click.echo("=" * 78)
    click.echo("  GeoVision — эксперимент с предобработкой (UAV vs Sentinel, честный тест)")
    click.echo("=" * 78)
    click.echo(f"Query: {query_path.name}")
    click.echo(f"Патчей в базе: {len(patch_files)}  |  Truth: {truth_file}")
    click.echo("")

    normalize_opts = [False, True] if sweep_normalize else [False]
    clahe_opts = [False, True] if sweep_clahe else [False]
    lcn_opts = [False, True] if sweep_lcn else [False]

    header = f"{'scale':>6} {'norm':>5} {'clahe':>6} {'lcn':>5} {'q_kp':>6} {'rank':>6} {'inliers':>8} {'time_s':>7}"
    click.echo(header)
    click.echo("-" * len(header))

    all_results = []
    for scale, norm, clahe, lcn in itertools.product(scales, normalize_opts, clahe_opts, lcn_opts):
        res = run_config(
            query_path, patch_files, truth_file,
            resize_scale=scale, normalize=norm, use_clahe=clahe, use_lcn=lcn,
            lowe_ratio=lowe_ratio, ransac_threshold=ransac_threshold, min_good_matches=min_good_matches,
        )
        if "error" in res:
            click.echo(f"{scale:>6} {norm!s:>5} {clahe!s:>6} {lcn!s:>5}   -- {res['error']} --")
            continue

        rank_str = str(res["rank"]) if res["rank"] else "NOT FOUND"
        line = (
            f"{scale:>6} {norm!s:>5} {clahe!s:>6} {lcn!s:>5} "
            f"{res['n_query_kp']:>6} {rank_str:>6} {res['truth_inliers']:>8} {res['elapsed_s']:>7}"
        )
        click.echo(line)
        all_results.append({"scale": scale, "normalize": norm, "clahe": clahe, "lcn": lcn, **res})

    click.echo("")
    valid = [r for r in all_results if r.get("rank")]
    if valid:
        best = min(valid, key=lambda r: r["rank"])
        click.secho(
            f"Лучшая конфигурация: scale={best['scale']} normalize={best['normalize']} "
            f"clahe={best['clahe']} lcn={best['lcn']} → rank={best['rank']} "
            f"(inliers={best['truth_inliers']}, было бы top-{best['n_candidates']})",
            fg="green",
        )
    else:
        click.secho("Ни в одной конфигурации правильный патч не найден среди кандидатов.", fg="red")


if __name__ == "__main__":
    main()
