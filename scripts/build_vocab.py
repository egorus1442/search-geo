#!/usr/bin/env python
"""
CLI скрипт: обучить BoVW словарь + построить FAISS индекс.

Запускать после того как ingestion завершён.

Пример:
    python scripts/build_vocab.py
    python scripts/build_vocab.py --only-index   # если словарь уже готов
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click

from config import configure_logging, get_logger
from workers.tasks.index_task import build_vocabulary, build_index

logger = get_logger(__name__)


@click.command()
@click.option("--only-index", is_flag=True, default=False, help="Пропустить обучение словаря")
@click.option("--only-vocab", is_flag=True, default=False, help="Только словарь, без индекса")
@click.option("--log-level", default="INFO", show_default=True)
def main(only_index, only_vocab, log_level):
    configure_logging(log_level)

    if not only_index:
        click.echo("Step 1: Building BoVW vocabulary...")
        result = build_vocabulary.apply().get()
        click.echo(f"  Vocabulary done: {result}")

    if not only_vocab:
        click.echo("Step 2: Building FAISS index...")
        result = build_index.apply().get()
        click.echo(f"  Index done: {result}")

    click.echo("\nAll done. Start the API and call POST /api/v1/localize")


if __name__ == "__main__":
    main()
