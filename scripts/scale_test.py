#!/usr/bin/env python
"""
Тест масштабируемости coarse-стадии (VLAD) до площадей ~1.5 млн км².

Что измеряем и почему
---------------------
Реальную базу в 1.5 млн км² (~5 млн патчей, ~100+ ГБ Sentinel-2, дни скачивания)
собрать в рамках сессии нельзя. Но главный вопрос масштаба — не «влезет ли»,
а два конкретных:

  1. RECALL при росте базы: остаётся ли ИСТИННЫЙ патч (и его соседи по селу)
     в пределах top_n_coarse, когда вокруг появляются миллионы других патчей?
     Если истинный патч на малой базе стоит на ранге R из N, то доля «ложно
     более похожих» = R/N. При росте базы в K раз этих ложных в среднем тоже
     ~в K раз больше — вот главный риск, что верный кандидат уйдёт за top-100.

  2. LATENCY: справляется ли FAISS по скорости на миллионах векторов.

Методика
--------
RECALL меряем ТОЧНО и без OOM: считаем расстояние query до «хороших» патчей
(село Багаряк), затем ПОТОКОВО генерируем синтетические дистракторы и считаем,
сколько из них ближе к query, чем хороший патч. Итоговый ранг = 1 + (реальных
ближе) + (дистракторов ближе). Так получаем точный ранг среди миллионов при
O(batch) памяти.

Дистракторы генерируются НА МНОГООБРАЗИИ реальных VLAD-векторов (выпуклые
комбинации случайных пар реальных патчей + шум, L2-норм) — то есть той же
природы, что реальные патчи региона. Это осознанно ПЕССИМИСТИЧНАЯ модель: она
предполагает, что вся страна статистически похожа на окрестности Багаряка;
реальные удалённые регионы отличаются сильнее и отсеиваются легче. Поэтому
измеренный ранг — верхняя (худшая) оценка; реальный будет не хуже.

LATENCY меряем на настоящем FAISS IVF-PQ индексе, набитом до целевого размера
(IVF-PQ — именно то, что применяют на миллионах, см. faiss_store.py).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import faiss
import numpy as np
from geoalchemy2.shape import to_shape

from config import configure_logging
from services.db.models import Patch
from services.db.session import SyncSessionLocal
from services.features.sift import extract_patch_descriptors, extract_query_descriptors
from services.features.vlad import VladEncoder
from services.ingestor.storage import download_bytes

CACHE = Path("/data/index/scale_cache.npz")


def _load_real_vectors(query_path: str):
    enc = VladEncoder.load()

    with SyncSessionLocal() as session:
        rows = session.query(Patch).all()
        patches = []
        for p in rows:
            c = to_shape(p.center)
            patches.append((p.id, p.s3_path, c.y, c.x))
    patches.sort(key=lambda x: x[0])

    if CACHE.exists():
        data = np.load(CACHE, allow_pickle=True)
        V, ids, lat, lon, q = data["V"], data["ids"], data["lat"], data["lon"], data["q"]
        click.echo(f"loaded cached vectors: {V.shape}")
        return enc, V, ids, lat, lon, q

    click.echo(f"encoding {len(patches)} real patches (VLAD)...")
    V, ids, lat, lon = [], [], [], []
    for i, (pid, s3, la, lo) in enumerate(patches):
        if i % 500 == 0:
            click.echo(f"  {i}/{len(patches)}")
        try:
            _, d = extract_patch_descriptors(download_bytes(s3))
            if d is None:
                continue
            V.append(enc.encode(d))
            ids.append(pid); lat.append(la); lon.append(lo)
        except Exception as exc:
            click.echo(f"  warn {pid}: {exc}", err=True)

    V = np.vstack(V).astype(np.float32)
    ids = np.array(ids); lat = np.array(lat); lon = np.array(lon)

    _, qd = extract_query_descriptors(Path(query_path).read_bytes())
    q = enc.encode(qd).astype(np.float32)

    np.savez(CACHE, V=V, ids=ids, lat=lat, lon=lon, q=q)
    click.echo(f"cached vectors: {V.shape}")
    return enc, V, ids, lat, lon, q


def _gen_distractors(V: np.ndarray, n: int, rng: np.random.Generator, noise: float) -> np.ndarray:
    """n синтетических VLAD-векторов на многообразии реальных (выпуклая комбинация
    случайных пар + гаусс. шум, L2-нормировка)."""
    m = len(V)
    i = rng.integers(0, m, size=n)
    j = rng.integers(0, m, size=n)
    a = rng.random((n, 1), dtype=np.float32)
    out = a * V[i] + (1.0 - a) * V[j]
    out += rng.normal(0.0, noise, size=out.shape).astype(np.float32)
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out.astype(np.float32)


@click.command()
@click.option("--query", default="bagaryak_query.jpg", show_default=True)
@click.option("--good-ids", default="1441,1489,1488", show_default=True,
              help="patch_id хороших патчей села (эталон корректной локализации)")
@click.option("--target-patches", default=5_000_000, show_default=True,
              help="целевой размер базы (~1.5 млн км² при ~0.3 км²/патч ≈ 5 млн)")
@click.option("--steps", default="3008,30000,300000,3000000,5000000", show_default=True,
              help="контрольные размеры базы для rank-кривой")
@click.option("--batch", default=250_000, show_default=True)
@click.option("--noise", default=0.02, show_default=True)
@click.option("--top-n-coarse", default=100, show_default=True)
@click.option("--latency/--no-latency", default=True, show_default=True,
              help="строить реальный FAISS IVF-PQ на target-patches и мерить время запроса")
@click.option("--log-level", default="WARNING", show_default=True)
def main(query, good_ids, target_patches, steps, batch, noise, top_n_coarse, latency, log_level):
    configure_logging(log_level)
    rng = np.random.default_rng(42)

    good = [int(x) for x in good_ids.split(",") if x.strip()]
    step_sizes = sorted({int(x) for x in steps.split(",") if x.strip()})

    enc, V, ids, lat, lon, q = _load_real_vectors(query)
    n_real = len(V)
    dim = V.shape[1]

    # ── Baseline: расстояния query до всех реальных + до хороших ──────────────────
    d_real = np.linalg.norm(V - q, axis=1)  # (n_real,)
    id_to_row = {int(pid): r for r, pid in enumerate(ids)}
    good_rows = {g: id_to_row[g] for g in good if g in id_to_row}
    good_d = {g: float(d_real[r]) for g, r in good_rows.items()}

    click.echo("")
    click.echo(f"real patches={n_real} dim={dim} query_dist(min/median)="
               f"{d_real.min():.4f}/{np.median(d_real):.4f}")
    click.echo("good patches baseline (среди реальных):")
    for g, r in good_rows.items():
        rank = int((d_real < good_d[g]).sum()) + 1
        click.echo(f"  patch {g}: dist={good_d[g]:.4f} rank={rank}/{n_real} "
                   f"lat={lat[r]:.5f} lon={lon[r]:.5f}")

    # «реальных ближе» для каждого хорошего патча (константа)
    real_closer = {g: int((d_real < good_d[g]).sum()) for g in good_rows}

    # ── Rank-кривая: сколько дистракторов ближе, чем хороший патч ─────────────────
    max_distractors = max(step_sizes) - n_real
    max_distractors = max(0, max_distractors)
    click.echo("")
    click.echo(f"streaming up to {max_distractors:,} synthetic on-manifold distractors "
               f"(pessimistic: вся страна ~как Багаряк)...")

    # накопительный счётчик дистракторов, которые ближе хорошего патча
    dist_closer = {g: 0 for g in good_rows}
    produced = 0
    # контрольные размеры базы → нужное число дистракторов
    checkpoints = [(s, s - n_real) for s in step_sizes if s > n_real]
    ck_idx = 0
    curve = {g: [] for g in good_rows}  # (base_size, rank)

    # ранг на базе = только реальные (step == n_real)
    for s in step_sizes:
        if s <= n_real:
            for g in good_rows:
                curve[g].append((s, real_closer[g] + 1))

    t0 = time.time()
    while produced < max_distractors:
        b = min(batch, max_distractors - produced)
        D = _gen_distractors(V, b, rng, noise)
        dd = np.linalg.norm(D - q, axis=1)
        for g in good_rows:
            dist_closer[g] += int((dd < good_d[g]).sum())
        produced += b

        while ck_idx < len(checkpoints) and produced >= checkpoints[ck_idx][1]:
            base_size = checkpoints[ck_idx][0]
            for g in good_rows:
                rank = real_closer[g] + dist_closer[g] + 1
                curve[g].append((base_size, rank))
            ck_idx += 1
    click.echo(f"  distractor streaming done in {time.time()-t0:.1f}s")

    # ── Печать rank-кривой ───────────────────────────────────────────────────────
    click.echo("")
    click.echo("RANK истинных патчей при росте базы (rank <= top_n_coarse → доживёт до RANSAC):")
    header = f"{'base_size':>12} " + " ".join(f"p{g:>7}" for g in good_rows)
    click.echo(header)
    click.echo("-" * len(header))
    all_sizes = sorted({s for g in good_rows for s, _ in curve[g]})
    for s in all_sizes:
        cells = []
        for g in good_rows:
            rank = dict(curve[g]).get(s)
            mark = "" if rank is None else ("*" if rank <= top_n_coarse else " ")
            cells.append(f"{rank}{mark}" if rank is not None else "-")
        click.echo(f"{s:>12,} " + " ".join(f"{c:>8}" for c in cells))
    click.echo(f"(* = в пределах top_n_coarse={top_n_coarse}; хотя бы один * = локализация ещё возможна)")

    # ── LATENCY: реальный FAISS IVF-PQ на target размере ─────────────────────────
    if latency:
        click.echo("")
        n_target = target_patches
        n_lists = 4096
        m_pq = 32  # подкванторов; dim=256 делится на 32
        click.echo(f"building FAISS IVF-PQ: n={n_target:,} dim={dim} nlist={n_lists} m={m_pq} ...")
        quant = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFPQ(quant, dim, n_lists, m_pq, 8)

        train_n = min(200_000, n_target)
        train = _gen_distractors(V, train_n, rng, noise)
        t0 = time.time()
        index.train(train)
        click.echo(f"  trained on {train_n:,} in {time.time()-t0:.1f}s")
        del train

        # добавляем реальные + дистракторы батчами
        index.add(V)
        added = n_real
        t0 = time.time()
        while added < n_target:
            b = min(batch, n_target - added)
            index.add(_gen_distractors(V, b, rng, noise))
            added += b
        click.echo(f"  added {added:,} vectors in {time.time()-t0:.1f}s")

        for nprobe in (16, 64, 256):
            index.nprobe = nprobe
            # прогрев
            index.search(q.reshape(1, -1), top_n_coarse)
            t0 = time.time()
            reps = 20
            for _ in range(reps):
                index.search(q.reshape(1, -1), top_n_coarse)
            ms = (time.time() - t0) / reps * 1000
            click.echo(f"  nprobe={nprobe:>3}: {ms:.1f} ms/query (top-{top_n_coarse}, base={n_target:,})")

    click.echo("")
    click.echo("Вывод: latency показывает скорость FAISS на целевом масштабе; "
               "rank-кривая — доживает ли верный патч до верификации.")


if __name__ == "__main__":
    main()
