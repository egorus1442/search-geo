"""
Celery task: построить/перестроить FAISS индекс по всем патчам в БД.

Очередь: index
"""
import io

import numpy as np

from workers.celery_app import app
from config import get_logger, get_settings
from services.db.session import SyncSessionLocal
from services.features.sift import extract_patch_descriptors
from services.features.coarse import load_coarse_encoder, new_coarse_encoder
from services.index.faiss_store import FaissStore
from services.index.metadata_store import PatchRepo
from services.ingestor.storage import download_bytes

logger = get_logger(__name__)
_s = get_settings()


@app.task(
    bind=True,
    name="workers.tasks.index_task.build_vocabulary",
    queue="index",
    time_limit=7200,
)
def build_vocabulary(self) -> dict:
    """
    Шаг 1: обучить coarse-энкодер (BoVW словарь или VLAD, см. COARSE_METHOD)
    на SIFT-дескрипторах из базы патчей. Сохраняет модель в /data/index/.

    VLAD проходит по базе дважды (k-means центроидов, затем PCA-whitening),
    поэтому дескрипторы отдаются через фабрику потоков (re-iterable).
    """
    method = _s.coarse_method.lower()
    logger.info("coarse_build_start", method=method)

    with SyncSessionLocal() as session:
        repo = PatchRepo(session)
        patch_ids = repo.get_all_patch_ids()

    n_total = len(patch_ids)
    if n_total == 0:
        raise ValueError("No patches in database. Run ingest first.")

    logger.info("coarse_patch_count", n=n_total, method=method)

    def descriptor_stream():
        for i, pid in enumerate(patch_ids):
            if i % 500 == 0:
                logger.info("coarse_stream_progress", i=i, total=n_total)
                self.update_state(state="PROGRESS", meta={"step": "coarse", "progress": i / n_total})
            try:
                with SyncSessionLocal() as session:
                    repo = PatchRepo(session)
                    meta_list = repo.get_patches_by_ids([pid])
                if not meta_list:
                    continue
                s3_path = meta_list[0]["s3_path"]
                img_bytes = download_bytes(s3_path)
                _, descs = extract_patch_descriptors(img_bytes)
                if descs is not None:
                    yield descs
            except Exception as exc:
                logger.warning("coarse_stream_error", patch_id=pid, error=str(exc))

    encoder = new_coarse_encoder(method)
    # VLAD ждёт фабрику потоков (двойной проход), BoVW — одиночный генератор.
    if method == "vlad":
        encoder.fit(descriptor_stream)  # type: ignore[arg-type]
    else:
        encoder.fit(descriptor_stream())  # type: ignore[call-arg]
    path = encoder.save()

    logger.info("coarse_build_done", path=str(path), method=method, dim=encoder.dim)
    return {"status": "done", "coarse_method": method, "model_path": str(path), "dim": encoder.dim}


@app.task(
    bind=True,
    name="workers.tasks.index_task.build_index",
    queue="index",
    time_limit=7200,
)
def build_index(self) -> dict:
    """
    Шаг 2: закодировать все патчи в coarse-векторы (BoVW или VLAD) и построить
    FAISS индекс. Требует предварительно обученного энкодера (build_vocabulary).
    """
    logger.info("index_build_start", method=_s.coarse_method)

    vocab = load_coarse_encoder()

    with SyncSessionLocal() as session:
        repo = PatchRepo(session)
        patch_ids = repo.get_all_patch_ids()

    n_total = len(patch_ids)
    logger.info("index_patch_count", n=n_total)

    all_hists: list[np.ndarray] = []
    all_ids: list[int] = []

    batch_size = 256
    for batch_start in range(0, n_total, batch_size):
        batch_ids = patch_ids[batch_start: batch_start + batch_size]

        progress = batch_start / n_total
        self.update_state(state="PROGRESS", meta={"step": "encode", "progress": progress})

        with SyncSessionLocal() as session:
            repo = PatchRepo(session)
            meta_list = repo.get_patches_by_ids(batch_ids)

        for meta in meta_list:
            try:
                img_bytes = download_bytes(meta["s3_path"])
                _, descs = extract_patch_descriptors(img_bytes)
                hist = vocab.encode(descs)
                all_hists.append(hist)
                all_ids.append(meta["patch_id"])
            except Exception as exc:
                logger.warning("index_encode_error", patch_id=meta["patch_id"], error=str(exc))

        if batch_start % (batch_size * 10) == 0:
            logger.info("index_encode_progress", done=len(all_hists), total=n_total)

    if not all_hists:
        raise ValueError("No histograms computed. Check vocabulary and patches.")

    vectors = np.vstack(all_hists).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int64)

    store = FaissStore()
    store.build_from_scratch(vectors, ids)
    path = store.save()

    logger.info("index_build_done", ntotal=store.ntotal, path=str(path))
    return {"status": "done", "ntotal": store.ntotal, "index_path": str(path)}


@app.task(
    name="workers.tasks.index_task.rebuild_all",
    queue="index",
)
def rebuild_all() -> dict:
    """Полный перестрой: словарь → индекс (запускает синхронно в том же процессе)."""
    vocab_result = build_vocabulary.apply().get()
    index_result = build_index.apply().get()
    return {"vocabulary": vocab_result, "index": index_result}
