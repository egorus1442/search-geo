#!/usr/bin/env python
"""
Комбинированный тест по Esri-базе: две стадии на одной выборке патчей.

  1. COARSE (dino_vlad) — «большой отсев» нейросетью: на каком ранге из ВСЕХ
     патчей стоит истинный тайл (то, что в бою делает FAISS).
  2. VERIFIER (SIFT+RANSAC) на top-K coarse-кандидатах — финальный ранг
     истинного тайла после геометрической привязки.

Не трогает боевой FAISS-индекс и не перестраивает базу: coarse-энкодер
обучается на лету на отфильтрованной подвыборке (--product-contains). Это
позволяет мерить Esri-базу, не удаляя старую Sentinel-базу.

Пример:
    python scripts/eval_esri_test.py \
        --query esri_bagaryak_wide.jpg \
        --center-lon 61.5172 --center-lat 56.2052 \
        --product-contains esri --patch-size 700 --top-k 300
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

from config import configure_logging, get_settings
from services.db.models import Patch
from services.db.session import SyncSessionLocal
from services.features.sift import (
    extract_patch_descriptors,
    extract_patch_gray,
    extract_query_descriptors,
    extract_query_gray,
)
from services.features.dino import DinoVladEncoder
from services.matching.verifier import Verifier
from services.ingestor.storage import download_bytes


def _hav_km(lon1, lat1, lon2, lat2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@click.command()
@click.option("--query", required=True, type=click.Path(exists=True))
@click.option("--center-lon", type=float, required=True)
@click.option("--center-lat", type=float, required=True)
@click.option("--truth-tol-km", default=1.0, show_default=True)
@click.option("--product-contains", default="esri", show_default=True)
@click.option("--patch-size", default=None, type=int)
@click.option("--top-k", default=300, show_default=True, help="coarse-кандидатов в verifier")
@click.option("--log-level", default="WARNING", show_default=True)
def main(query, center_lon, center_lat, truth_tol_km, product_contains, patch_size, top_k, log_level):
    configure_logging(log_level)
    s = get_settings()

    # ── кандидаты из БД (фильтр по источнику/размеру) ────────────────────────────
    with SyncSessionLocal() as session:
        rows = session.query(Patch).all()
        cands = []
        for p in rows:
            if product_contains and (not p.source_tile or product_contains not in p.source_tile.product_id):
                continue
            if patch_size is not None and p.patch_size != patch_size:
                continue
            c = to_shape(p.center)
            bb = to_shape(p.bbox).bounds
            cands.append({
                "patch_id": p.id,
                "center_lon": c.x,
                "center_lat": c.y,
                "bbox": [bb[0], bb[1], bb[2], bb[3]],
                "s3_path": p.s3_path,
            })
    if not cands:
        raise click.ClickException("нет патчей после фильтров")

    # ── истинный патч = ближайший центр к точке ──────────────────────────────────
    nearest = min(cands, key=lambda c: _hav_km(center_lon, center_lat, c["center_lon"], c["center_lat"]))
    d0 = _hav_km(center_lon, center_lat, nearest["center_lon"], nearest["center_lat"])
    if d0 > truth_tol_km:
        raise click.ClickException(f"ближайший патч {d0:.2f} км > tol={truth_tol_km}")
    truth_id = nearest["patch_id"]
    click.echo(f"patches={len(cands)}  truth patch_id={truth_id} ({d0*1000:.0f} м от точки)")

    # ── дескрипторы query ────────────────────────────────────────────────────────
    qbytes = Path(query).read_bytes()
    qkp, qdesc = extract_query_descriptors(qbytes)
    qgray = extract_query_gray(qbytes) if s.photometric_check_enabled else None
    click.echo(
        f"query features={0 if qdesc is None else len(qdesc)}  "
        f"query_resize={s.preprocess_query_resize_scale}  patch_resize={s.preprocess_patch_resize_scale}"
    )

    click.echo("downloading patches from MinIO...")
    bytes_by = {c["patch_id"]: download_bytes(c["s3_path"]) for c in cands}

    # ── COARSE: dino_vlad ────────────────────────────────────────────────────────
    click.echo("fitting dino_vlad + encoding (coarse, большой отсев)...")
    t0 = time.time()
    enc = DinoVladEncoder()
    enc.fit_images(lambda: (bytes_by[c["patch_id"]] for c in cands))
    qv = enc.encode_image(qbytes)
    dists = []
    for c in cands:
        v = enc.encode_image(bytes_by[c["patch_id"]])
        dists.append((float(np.linalg.norm(qv - v)), c["patch_id"]))
    dists.sort(key=lambda x: x[0])
    order = [pid for _, pid in dists]
    coarse_rank = order.index(truth_id) + 1
    n = len(order)

    def at(k):
        return "yes" if coarse_rank <= k else "no"

    click.echo("")
    click.echo(f"=== COARSE (dino_vlad, dim={enc.dim}) — нейронка, большой отсев ===")
    click.echo(
        f"truth coarse rank = {coarse_rank} / {n}   "
        f"R@10={at(10)} R@50={at(50)} R@100={at(100)} R@300={at(300)}   ({time.time()-t0:.0f}s)"
    )

    # ── VERIFIER на top-K ────────────────────────────────────────────────────────
    id2c = {c["patch_id"]: c for c in cands}
    topk_cands = [id2c[i] for i in order[:top_k]]
    path_bytes = {c["s3_path"]: bytes_by[c["patch_id"]] for c in topk_cands}

    def load_desc(s3_path):
        raw = path_bytes[s3_path]
        kp, desc = extract_patch_descriptors(raw)
        gray = extract_patch_gray(raw) if s.photometric_check_enabled else None
        return kp, desc, gray

    click.echo("")
    click.echo(f"=== VERIFIER (SIFT+RANSAC) на top-{top_k} coarse-кандидатах ===")
    t1 = time.time()
    ver = Verifier(top_n=len(topk_cands))
    results = ver.verify(
        query_kp=qkp, query_desc=qdesc, candidates=topk_cands,
        load_desc_fn=load_desc, query_gray=qgray,
    )
    final_order = [r.patch_id for r in results]
    if truth_id in final_order:
        fr = final_order.index(truth_id) + 1
        tr = results[fr - 1]
        click.echo(
            f"truth FINAL rank = {fr} / {len(results)} verified   "
            f"inliers={tr.inlier_count} ratio={tr.inlier_ratio} score={tr.score}"
        )
    else:
        click.echo(f"truth НЕ прошёл verify (0 inliers). verified={len(results)}")

    click.echo("top-5 verified:")
    for i, r in enumerate(results[:5], 1):
        mark = "*" if r.patch_id == truth_id else " "
        derr = _hav_km(center_lon, center_lat, r.center_lon, r.center_lat) * 1000
        click.echo(
            f"  {i}{mark} patch_id={r.patch_id} inliers={r.inlier_count} "
            f"ratio={r.inlier_ratio} conf={r.confidence} err={derr:.0f}m"
        )
    click.echo(f"verify time {time.time()-t1:.0f}s")


if __name__ == "__main__":
    main()
