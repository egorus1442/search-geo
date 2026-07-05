"""
Celery task: построить/перестроить FAISS индекс по всем патчам в БД.

Очередь: index
"""
import io

import numpy as np

from workers.celery_app import app
from config import get_logger
from services.db.session import SyncSessionLocal
from services.features.sift import extract_patch_descriptors
from services.features.vocabulary import Vocabulary
from services.index.faiss_store import FaissStore
from services.index.metadata_store import PatchRepo
from services.ingestor.storage import download_bytes

logger = get_logger(__name__)


@app.task(
    bind=True,
    name="workers.tasks.index_task.build_vocabulary",
    queue="index",
    time_limit=7200,
)
def build_vocabulary(self) -> dict:
    """
    Шаг 1: обучить BoVW словарь на SIFT дескрипторах из базы патчей.
    Сохраняет vocabulary.pkl в /data/index/.
    """
    logger.info("vocab_build_start")

    with SyncSessionLocal() as session:
        repo = PatchRepo(session)
        patch_ids = repo.get_all_patch_ids()

    n_total = len(patch_ids)
    if n_total == 0:
        raise ValueError("No patches in database. Run ingest first.")

    logger.info("vocab_patch_count", n=n_total)

    def descriptor_stream():
        for i, pid in enumerate(patch_ids):
            if i % 500 == 0:
                logger.info("vocab_stream_progress", i=i, total=n_total)
                self.update_state(state="PROGRESS", meta={"step": "vocab", "progress": i / n_total})
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
                logger.warning("vocab_stream_error", patch_id=pid, error=str(exc))

    vocab = Vocabulary()
    vocab.fit(descriptor_stream())
    path = vocab.save()

    logger.info("vocab_build_done", path=str(path))
    return {"status": "done", "vocab_path": str(path), "vocab_size": vocab.vocab_size}


@app.task(
    bind=True,
    name="workers.tasks.index_task.build_index",
    queue="index",
    time_limit=7200,
)
def build_index(self) -> dict:
    """
    Шаг 2: закодировать все патчи в BoVW гистограммы и построить FAISS индекс.
    Требует предварительно обученного словаря (build_vocabulary).
    """
    logger.info("index_build_start")

    vocab = Vocabulary.load()

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
