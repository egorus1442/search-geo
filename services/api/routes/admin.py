import uuid

from fastapi import APIRouter, HTTPException, status

from config import get_logger
from services.api.schemas import IndexStatsResponse, IngestRequest, TaskResponse
from services.db.session import SyncSessionLocal
from services.index.faiss_store import FaissStore
from services.index.metadata_store import PatchRepo, TaskRepo
from services.features.vocabulary import Vocabulary
from services.matching.localize import reload_indexes
from workers.tasks.ingest_task import run_ingest
from workers.tasks.index_task import build_vocabulary, build_index

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])
logger = get_logger(__name__)


# ── Ingest ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def start_ingest(req: IngestRequest):
    """Запустить скачивание Sentinel-2 снимков и нарезку на патчи."""
    with SyncSessionLocal() as session:
        task_repo = TaskRepo(session)
        task = task_repo.create(
            task_type="ingest",
            params={
                "bbox": req.bbox,
                "date_from": req.date_from,
                "date_to": req.date_to,
                "cloud_cover_max": req.cloud_cover_max,
            },
        )
        task_id = str(task.id)

    celery_task = run_ingest.apply_async(
        kwargs={
            "bbox": req.bbox,
            "date_from": req.date_from,
            "date_to": req.date_to,
            "cloud_cover_max": req.cloud_cover_max,
            "task_db_id": task_id,
        },
        queue="ingest",
    )

    with SyncSessionLocal() as session:
        TaskRepo(session).update_status(
            uuid.UUID(task_id), "running", celery_id=celery_task.id
        )

    logger.info("ingest_task_started", task_id=task_id, celery_id=celery_task.id)
    return TaskResponse(task_id=task_id, status="running", type="ingest")


@router.get("/ingest/{task_id}", response_model=TaskResponse)
def get_ingest_status(task_id: str):
    """Статус задачи ingestion."""
    try:
        tid = uuid.UUID(task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format")

    with SyncSessionLocal() as session:
        task = TaskRepo(session).get(tid)

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse(
        task_id=str(task.id),
        status=task.status,
        type=task.type,
        result=task.result,
        error=task.error,
    )


# ── Index ─────────────────────────────────────────────────────────────────────

@router.post("/index/vocabulary", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def start_build_vocabulary():
    """Запустить обучение BoVW словаря (шаг 1 индексирования)."""
    celery_task = build_vocabulary.apply_async(queue="index")
    return TaskResponse(task_id=celery_task.id, status="running", type="build_vocabulary")


@router.post("/index/build", response_model=TaskResponse, status_code=status.HTTP_202_ACCEPTED)
def start_build_index():
    """Построить FAISS индекс (шаг 2, после словаря)."""
    celery_task = build_index.apply_async(queue="index")
    return TaskResponse(task_id=celery_task.id, status="running", type="build_index")


@router.post("/index/reload")
def reload_index():
    """Перезагрузить индекс и словарь из файлов (после rebuild)."""
    try:
        reload_indexes()
        return {"status": "ok", "message": "Indexes reloaded"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/index/stats", response_model=IndexStatsResponse)
def index_stats():
    """Статистика FAISS индекса и базы патчей."""
    try:
        store = FaissStore.load()
        stats = store.stats()
    except FileNotFoundError:
        stats = {"ntotal": 0, "dim": 0, "n_lists": 0, "n_probe": 0, "is_trained": False}

    try:
        vocab = Vocabulary.load()
        vocab_size = vocab.vocab_size
    except FileNotFoundError:
        vocab_size = 0

    with SyncSessionLocal() as session:
        patch_count = PatchRepo(session).count_patches()

    return IndexStatsResponse(
        faiss_ntotal=stats.get("ntotal", 0),
        faiss_dim=stats.get("dim", 0),
        faiss_n_lists=stats.get("n_lists", 0),
        faiss_n_probe=stats.get("n_probe", 0),
        faiss_is_trained=stats.get("is_trained", False),
        vocab_size=vocab_size,
        patch_count_db=patch_count,
    )
