"""
Работа с PostgreSQL: патчи, их координаты, маппинг patch_id → geo.
"""
from __future__ import annotations

import uuid
from typing import Any

from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, box
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from services.db.models import Patch, SourceTile, Task
from config import get_logger

logger = get_logger(__name__)


class PatchRepo:
    """Синхронный репозиторий для Celery-задач."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create_source_tile(
        self,
        product_id: str,
        bbox: tuple[float, float, float, float],
        date_acq: Any,
        cloud_cover: float | None,
        s3_path: str | None = None,
    ) -> SourceTile:
        lon_min, lat_min, lon_max, lat_max = bbox
        geom = from_shape(box(lon_min, lat_min, lon_max, lat_max), srid=4326)
        tile = SourceTile(
            product_id=product_id,
            bbox=geom,
            date_acq=date_acq,
            cloud_cover=cloud_cover,
            s3_path=s3_path,
            status="downloaded",
        )
        self._session.add(tile)
        self._session.flush()
        return tile

    def tile_exists(self, product_id: str) -> bool:
        row = self._session.execute(
            select(SourceTile.id).where(SourceTile.product_id == product_id)
        ).first()
        return row is not None

    def create_patch(
        self,
        source_tile_id: uuid.UUID,
        center_lon: float,
        center_lat: float,
        bbox: tuple[float, float, float, float],
        s3_path: str,
        patch_size: int = 256,
        gsd_m: float = 10.0,
    ) -> Patch:
        lon_min, lat_min, lon_max, lat_max = bbox
        patch = Patch(
            source_tile_id=source_tile_id,
            center=from_shape(Point(center_lon, center_lat), srid=4326),
            bbox=from_shape(box(lon_min, lat_min, lon_max, lat_max), srid=4326),
            s3_path=s3_path,
            patch_size=patch_size,
            gsd_m=gsd_m,
        )
        self._session.add(patch)
        self._session.flush()
        return patch

    def get_all_patch_ids(self) -> list[int]:
        rows = self._session.execute(select(Patch.id).order_by(Patch.id)).all()
        return [r[0] for r in rows]

    def get_patches_by_ids(self, ids: list[int]) -> list[dict[str, Any]]:
        rows = self._session.execute(
            select(Patch).where(Patch.id.in_(ids))
        ).scalars().all()

        result = []
        for patch in rows:
            center = to_shape(patch.center)
            bbox_shape = to_shape(patch.bbox)
            bounds = bbox_shape.bounds  # (minx, miny, maxx, maxy)
            result.append({
                "patch_id": patch.id,
                "center_lon": center.x,
                "center_lat": center.y,
                "bbox": [bounds[0], bounds[1], bounds[2], bounds[3]],
                "s3_path": patch.s3_path,
            })
        return result

    def count_patches(self) -> int:
        row = self._session.execute(select(func.count(Patch.id))).scalar()
        return row or 0

    def mark_tile_processed(self, tile_id: uuid.UUID) -> None:
        tile = self._session.get(SourceTile, tile_id)
        if tile:
            tile.status = "processed"
            self._session.flush()


class TaskRepo:
    """Репозиторий для управления задачами."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, task_type: str, params: dict | None = None) -> Task:
        task = Task(type=task_type, status="pending", params=params)
        self._session.add(task)
        self._session.commit()
        self._session.refresh(task)
        return task

    def update_status(
        self,
        task_id: uuid.UUID,
        status: str,
        celery_id: str | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        task = self._session.get(Task, task_id)
        if task:
            task.status = status
            if celery_id is not None:
                task.celery_id = celery_id
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            self._session.commit()

    def get(self, task_id: uuid.UUID) -> Task | None:
        return self._session.get(Task, task_id)
