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
from services.features.coarse import (
    coarse_index_path,
    encoder_input_kind,
    is_image_method,
    load_coarse_encoder,
    new_coarse_encoder,
)
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
    Шаг 1: обучить/подготовить coarse-энкодер (см. COARSE_METHOD) и сохранить
    модель в /data/index/.

    - vlad — двойной проход по базе (k-means + PCA-whitening) → фабрика потоков;
    - bovw — одиночный генератор SIFT-дескрипторов;
    - dino — обучения нет (pretrained), просто сохраняем конфиг;
    - dino_vlad (AnyLoc) — один проход по КАРТИНКАМ (k-means DINOv2-токенов).
    """
    method = _s.coarse_method.lower()
    image_method = is_image_method(method)
    logger.info("coarse_build_start", method=method, image_method=image_method)

    with SyncSessionLocal() as session:
        repo = PatchRepo(session)
        patch_ids = repo.get_all_patch_ids()

    n_total = len(patch_ids)
    if n_total == 0:
        raise ValueError("No patches in database. Run ingest first.")

    logger.info("coarse_patch_count", n=n_total, method=method)

    def _iter_patch_bytes():
        """Общий проход по патчам: yield (pid, img_bytes) с прогрессом."""
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
                yield pid, download_bytes(meta_list[0]["s3_path"])
            except Exception as exc:
                logger.warning("coarse_stream_error", patch_id=pid, error=str(exc))

    def descriptor_stream():
        for _pid, img_bytes in _iter_patch_bytes():
            _, descs = extract_patch_descriptors(img_bytes)
            if descs is not None:
                yield descs

    def image_stream():
        for _pid, img_bytes in _iter_patch_bytes():
            yield img_bytes

    encoder = new_coarse_encoder(method)
    if image_method:
        # dino: fit_images — no-op; dino_vlad: один проход по картинкам (k-means).
        encoder.fit_images(image_stream)  # type: ignore[attr-defined]
    elif method == "vlad":
        encoder.fit(descriptor_stream)  # type: ignore[arg-type]  # фабрика (двойной проход)
    else:
        encoder.fit(descriptor_stream())  # type: ignore[call-arg]  # одиночный генератор
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
    Шаг 2: закодировать все патчи в coarse-векторы и построить FAISS индекс.
    Требует предварительно подготовленного энкодера (build_vocabulary).

    Диспетчеризация по input_kind энкодера:
      - descriptors (vlad/bovw) — SIFT-дескрипторы патча → encode(descs);
      - image (dino/dino_vlad)  — картинка патча → encode_image_batch (без SIFT).
    Нейро-методы пишут в ОТДЕЛЬНЫЙ FAISS (GLOBAL_INDEX_PATH) и на малой базе
    строят Flat-индекс (точный) — см. GLOBAL_USE_IVF_THRESHOLD.
    """
    method = _s.coarse_method.lower()
    vocab = load_coarse_encoder()
    image_method = encoder_input_kind(vocab) == "image"
    logger.info("index_build_start", method=method, image_method=image_method)

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

        if image_method:
            imgs: list[bytes] = []
            ids_batch: list[int] = []
            for meta in meta_list:
                try:
                    imgs.append(download_bytes(meta["s3_path"]))
                    ids_batch.append(meta["patch_id"])
                except Exception as exc:
                    logger.warning("index_encode_error", patch_id=meta["patch_id"], error=str(exc))
            if imgs:
                vecs = vocab.encode_image_batch(imgs)  # type: ignore[attr-defined]
                all_hists.extend(vecs)
                all_ids.extend(ids_batch)
        else:
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
        raise ValueError("No vectors computed. Check encoder and patches.")

    vectors = np.vstack(all_hists).astype(np.float32)
    ids = np.array(all_ids, dtype=np.int64)

    # Нейро-индекс держим отдельно; на малой базе — Flat (точный, без IVF).
    index_path = coarse_index_path(method)
    use_flat = image_method and len(vectors) < _s.global_use_ivf_threshold
    store = FaissStore(index_path=index_path, flat=use_flat)
    store.build_from_scratch(vectors, ids)
    path = store.save()

    logger.info("index_build_done", ntotal=store.ntotal, path=str(path), flat=use_flat)
    return {"status": "done", "ntotal": store.ntotal, "index_path": str(path), "flat": use_flat}


@app.task(
    name="workers.tasks.index_task.rebuild_all",
    queue="index",
)
def rebuild_all() -> dict:
    """Полный перестрой: словарь → индекс (запускает синхронно в том же процессе)."""
    vocab_result = build_vocabulary.apply().get()
    index_result = build_index.apply().get()
    return {"vocabulary": vocab_result, "index": index_result}
