#!/usr/bin/env python
"""Evaluate external UAV query against geographically bounded Sentinel patches."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
from geoalchemy2.shape import to_shape

from config import configure_logging
from services.db.models import Patch
from services.db.session import SyncSessionLocal
from services.features.sift import extract_descriptors
from services.features.vocabulary import Vocabulary
from services.ingestor.storage import download_bytes
from services.matching.verifier import Verifier


def _window(center_lon: float, center_lat: float, size_km: float) -> tuple[float, float, float, float]:
    half = size_km / 2.0
    lat_delta = half / 111.32
    lon_delta = half / (111.32 * math.cos(math.radians(center_lat)))
    return (
        center_lon - lon_delta,
        center_lat - lat_delta,
        center_lon + lon_delta,
        center_lat + lat_delta,
    )


def _patch_meta(patch: Patch) -> dict:
    center = to_shape(patch.center)
    bounds = to_shape(patch.bbox).bounds
    return {
        "patch_id": patch.id,
        "center_lon": center.x,
        "center_lat": center.y,
        "bbox": [bounds[0], bounds[1], bounds[2], bounds[3]],
        "s3_path": patch.s3_path,
    }


def _load_desc(s3_path: str):
    kp, desc = extract_descriptors(download_bytes(s3_path))
    return kp, desc, None  # gray=None: этот скрипт не тестирует photometric-фильтр


@click.command()
@click.option("--query", default="external_uav_kharkiv_botanic_query_wide_570m.jpg", show_default=True)
@click.option("--center-lon", default=36.2402455, show_default=True)
@click.option("--center-lat", default=50.028757, show_default=True)
@click.option("--sizes", default="3,10,30", show_default=True, help="Comma-separated window sizes in km")
@click.option("--bbox", default=None, help="Optional exact bbox: lon_min,lat_min,lon_max,lat_max")
@click.option("--expected", default="1754,1755", show_default=True, help="Comma-separated expected patch IDs")
@click.option("--product-contains", default=None, help="Only use patches whose source product_id contains this text")
@click.option("--patch-size", default=None, type=int, help="Only use patches with this patch_size")
@click.option("--fit-vocab", is_flag=True, default=False, help="Fit a temporary vocabulary on selected patches")
@click.option("--top-k", default=20, show_default=True)
@click.option("--verify-k", default=50, show_default=True)
@click.option("--log-level", default="WARNING", show_default=True)
def main(
    query,
    center_lon,
    center_lat,
    sizes,
    bbox,
    expected,
    product_contains,
    patch_size,
    fit_vocab,
    top_k,
    verify_k,
    log_level,
):
    configure_logging(log_level)

    query_bytes = Path(query).read_bytes()
    query_kp, query_desc = extract_descriptors(query_bytes)
    if query_desc is None:
        raise click.ClickException("No SIFT descriptors extracted from query")

    expected_ids = {int(x) for x in expected.split(",") if x.strip()}
    verifier = Verifier(top_n=10)
    desc_cache: dict[str, tuple[object, np.ndarray | None, object]] = {}

    def load_desc_cached(s3_path: str):
        if s3_path not in desc_cache:
            desc_cache[s3_path] = _load_desc(s3_path)
        return desc_cache[s3_path]

    with SyncSessionLocal() as session:
        rows = session.query(Patch).all()
        if product_contains is not None:
            rows = [p for p in rows if p.source_tile and product_contains in p.source_tile.product_id]
        if patch_size is not None:
            rows = [p for p in rows if p.patch_size == patch_size]
        patches = [_patch_meta(p) for p in rows]

    if fit_vocab:
        click.echo("fitting temporary vocabulary...")
        vocab = Vocabulary()
        vocab.fit((load_desc_cached(p["s3_path"])[1] for p in patches))
    else:
        vocab = Vocabulary.load()

    query_hist = vocab.encode(query_desc)

    click.echo(f"query={query} features={len(query_desc)} patches_total={len(patches)}")

    windows: list[tuple[str, tuple[float, float, float, float]]] = []
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            raise click.ClickException("--bbox must contain 4 comma-separated values")
        windows.append(("bbox", (parts[0], parts[1], parts[2], parts[3])))
    for size in [float(x) for x in sizes.split(",") if x.strip()]:
        windows.append((f"{size:g}x{size:g}km", _window(center_lon, center_lat, size)))

    for label, (lon_min, lat_min, lon_max, lat_max) in windows:
        subset = [
            p for p in patches
            if lon_min <= p["center_lon"] <= lon_max and lat_min <= p["center_lat"] <= lat_max
        ]

        scored = []
        for p in subset:
            try:
                _, desc = load_desc_cached(p["s3_path"])
                if desc is None:
                    continue
                hist = vocab.encode(desc)
                dist = float(np.linalg.norm(query_hist - hist))
                scored.append((dist, p))
            except Exception as exc:
                click.echo(f"warn patch_id={p['patch_id']} error={exc}", err=True)

        scored.sort(key=lambda x: x[0])
        ranked = [p for _, p in scored]
        ranks = {
            pid: next((i + 1 for i, p in enumerate(ranked) if p["patch_id"] == pid), None)
            for pid in expected_ids
        }

        verified = verifier.verify(
            query_kp=query_kp,
            query_desc=query_desc,
            candidates=ranked[:verify_k],
            load_desc_fn=load_desc_cached,
        )

        click.echo("")
        click.echo(
            f"window={label} "
            f"bbox=[{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}] "
            f"candidates={len(subset)} scored={len(scored)} expected_ranks={ranks}"
        )
        click.echo("coarse_top:")
        for i, (dist, p) in enumerate(scored[:top_k], start=1):
            mark = "*" if p["patch_id"] in expected_ids else " "
            click.echo(
                f"  {i:02d}{mark} patch_id={p['patch_id']} dist={dist:.4f} "
                f"lat={p['center_lat']:.6f} lon={p['center_lon']:.6f}"
            )
        click.echo("verified_top:")
        for i, p in enumerate(verified, start=1):
            mark = "*" if p.patch_id in expected_ids else " "
            click.echo(
                f"  {i:02d}{mark} patch_id={p.patch_id} inliers={p.inlier_count} ratio={p.inlier_ratio:.2f} "
                f"conf={p.confidence:.2f} lat={p.center_lat:.6f} lon={p.center_lon:.6f}"
            )


if __name__ == "__main__":
    main()
