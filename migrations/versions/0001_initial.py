"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-06-30
"""
from typing import Sequence, Union

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "source_tiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("product_id", sa.String(255), nullable=False, unique=True),
        sa.Column("bbox", geoalchemy2.types.Geometry("POLYGON", srid=4326), nullable=False),
        sa.Column("date_acq", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cloud_cover", sa.Float(), nullable=True),
        sa.Column("s3_path", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), server_default="downloaded"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_source_tiles_product_id", "source_tiles", ["product_id"])
    op.create_index(
        "ix_source_tiles_bbox",
        "source_tiles",
        ["bbox"],
        postgresql_using="gist",
    )

    op.create_table(
        "patches",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column(
            "source_tile_id",
            UUID(as_uuid=True),
            sa.ForeignKey("source_tiles.id"),
            nullable=False,
        ),
        sa.Column("center", geoalchemy2.types.Geometry("POINT", srid=4326), nullable=False),
        sa.Column("bbox", geoalchemy2.types.Geometry("POLYGON", srid=4326), nullable=False),
        sa.Column("s3_path", sa.Text(), nullable=False),
        sa.Column("patch_size", sa.Integer(), server_default="256"),
        sa.Column("gsd_m", sa.Float(), server_default="10.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index(
        "ix_patches_center",
        "patches",
        ["center"],
        postgresql_using="gist",
    )
    op.create_index(
        "ix_patches_bbox",
        "patches",
        ["bbox"],
        postgresql_using="gist",
    )
    op.create_index("ix_patches_source_tile_id", "patches", ["source_tile_id"])

    op.create_table(
        "tasks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("celery_id", sa.String(255), nullable=True),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), server_default="pending"),
        sa.Column("params", JSONB(), nullable=True),
        sa.Column("result", JSONB(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("ix_tasks_celery_id", "tasks", ["celery_id"])
    op.create_index("ix_tasks_status", "tasks", ["status"])


def downgrade() -> None:
    op.drop_table("tasks")
    op.drop_table("patches")
    op.drop_table("source_tiles")
