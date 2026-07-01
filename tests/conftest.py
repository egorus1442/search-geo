"""Общие фикстуры для тестов."""
import os

import pytest

# Переопределяем пути для тестов — не нужны реальные файлы
os.environ.setdefault("FAISS_INDEX_PATH", "/tmp/test_index.faiss")
os.environ.setdefault("VOCAB_PATH", "/tmp/test_vocab.pkl")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://geo:geo@localhost:5432/geovision_test")
os.environ.setdefault("DATABASE_URL_SYNC", "postgresql+psycopg2://geo:geo@localhost:5432/geovision_test")
os.environ.setdefault("CDSE_USERNAME", "test@example.com")
os.environ.setdefault("CDSE_PASSWORD", "test_password")
