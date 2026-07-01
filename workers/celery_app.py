from celery import Celery

from config import get_settings

_s = get_settings()

app = Celery(
    "geovision",
    broker=_s.celery_broker_url,
    backend=_s.celery_result_backend,
    include=[
        "workers.tasks.ingest_task",
        "workers.tasks.index_task",
    ],
)

app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "workers.tasks.ingest_task.*": {"queue": "ingest"},
        "workers.tasks.index_task.*": {"queue": "index"},
    },
)
