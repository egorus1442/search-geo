"""Coarse-ранг истинного тайла в БОЕВОМ FAISS-индексе (нейросетевой отсев).

Кодирует query боевым энкодером (dino_vlad), ищет по реальному global-индексу
и печатает, на каком месте из всех патчей стоит истинный тайл — ровно то, что
в /localize уходит в top-K на верификацию.
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from geoalchemy2.shape import to_shape

from services.db.session import SyncSessionLocal
from services.db.models import Patch
from services.features.coarse import coarse_index_path, load_coarse_encoder
from services.index.faiss_store import FaissStore


def _hav(lon1, lat1, lon2, lat2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@click.command()
@click.option("--query", required=True, type=click.Path(exists=True))
@click.option("--center-lon", type=float, required=True)
@click.option("--center-lat", type=float, required=True)
def main(query, center_lon, center_lat):
    with SyncSessionLocal() as session:
        rows = session.query(Patch).all()
        metas = []
        for p in rows:
            c = to_shape(p.center)
            metas.append((p.id, c.x, c.y))
    truth_id, tlon, tlat = min(metas, key=lambda m: _hav(center_lon, center_lat, m[1], m[2]))
    d0 = _hav(center_lon, center_lat, tlon, tlat)
    click.echo(f"patches={len(metas)}  truth patch_id={truth_id} ({d0:.0f} m от точки)")

    enc = load_coarse_encoder()
    store = FaissStore.load(path=coarse_index_path())
    qv = enc.encode_image(Path(query).read_bytes())
    dists, ids = store.search(qv, k=store.ntotal)
    ids = [int(x) for x in ids if x != -1]
    rank = ids.index(truth_id) + 1 if truth_id in ids else None

    click.echo(f"=== COARSE (dino_vlad, боевой FAISS, ntotal={store.ntotal}) ===")
    click.echo(f"truth coarse rank = {rank} / {store.ntotal}")
    for k in (10, 50, 100, 300):
        click.echo(f"  R@{k}: {'yes' if rank and rank <= k else 'no'}")


if __name__ == "__main__":
    main()
