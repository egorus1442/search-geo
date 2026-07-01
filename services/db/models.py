import uuid
from datetime import datetime, timezone

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


def utcnow():
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SourceTile(Base):
    """Исходный тайл Sentinel-2, скачанный из CDSE."""

    __tablename__ = "source_tiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    product_id = Column(String(255), nullable=False, unique=True, index=True)
    bbox = Column(Geometry("POLYGON", srid=4326), nullable=False)
    date_acq = Column(DateTime(timezone=True), nullable=False)
    cloud_cover = Column(Float, nullable=True)
    s3_path = Column(Text, nullable=True)
    status = Column(String(32), default="downloaded")  # downloaded | processed | failed
    created_at = Column(DateTime(timezone=True), default=utcnow)

    patches = relationship("Patch", back_populates="source_tile", lazy="select")


class Patch(Base):
    """Патч 256×256 пкс из спутникового тайла — единица эталонной базы."""

    __tablename__ = "patches"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    source_tile_id = Column(UUID(as_uuid=True), ForeignKey("source_tiles.id"), nullable=False)
    center = Column(Geometry("POINT", srid=4326), nullable=False)
    bbox = Column(Geometry("POLYGON", srid=4326), nullable=False)
    s3_path = Column(Text, nullable=False)
    patch_size = Column(Integer, default=256)
    gsd_m = Column(Float, default=10.0)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    source_tile = relationship("SourceTile", back_populates="patches")


class Task(Base):
    """Фоновая задача: ingestion или indexing."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    celery_id = Column(String(255), nullable=True, index=True)
    type = Column(String(32), nullable=False)    # ingest | index | localize
    status = Column(String(32), default="pending")  # pending | running | done | failed
    params = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
