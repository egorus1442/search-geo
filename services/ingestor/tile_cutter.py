"""
Нарезает Sentinel-2 .SAFE продукт на патчи 256×256 пкс.

Pipeline:
  .zip → распаковка → .SAFE/GRANULE/.../IMG_DATA/R10m/*.jp2
  → rasterio warp (если нужно) → нарезка на патчи → PNG → MinIO
"""
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import Window
from PIL import Image

from config import get_logger, get_settings
from services.ingestor.storage import upload_bytes

logger = get_logger(__name__)
_s = get_settings()

# Sentinel-2 L2A RGB каналы в R10m
_S2_BANDS = {"R": "B04", "G": "B03", "B": "B02"}


@dataclass
class PatchMeta:
    patch_id: str
    s3_key: str
    center_lon: float
    center_lat: float
    bbox: tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    patch_size: int
    gsd_m: float


def extract_safe(zip_path: Path, target_dir: Path) -> Path:
    """Распаковать .zip → .SAFE директория."""
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)
    # Найти .SAFE директорию
    safe_dirs = list(target_dir.glob("*.SAFE"))
    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE directory found in {zip_path}")
    return safe_dirs[0]


def find_band_files(safe_dir: Path) -> dict[str, Path]:
    """Найти .jp2 файлы для RGB каналов в R10m."""
    band_paths: dict[str, Path] = {}
    for color, band_name in _S2_BANDS.items():
        pattern = f"GRANULE/*/IMG_DATA/R10m/*_{band_name}_10m.jp2"
        matches = list(safe_dir.glob(pattern))
        if not matches:
            # Попробовать без R10m суффикса (L1C)
            pattern = f"GRANULE/*/IMG_DATA/*_{band_name}.jp2"
            matches = list(safe_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"Band {band_name} not found in {safe_dir}")
        band_paths[color] = matches[0]
    return band_paths


def _normalize_band(data: np.ndarray) -> np.ndarray:
    """Привести к uint8 с насыщением на 2%/98% перцентилях."""
    p2, p98 = np.percentile(data[data > 0], (2, 98)) if (data > 0).any() else (0, 1)
    clipped = np.clip(data, p2, p98)
    if p98 > p2:
        normalized = ((clipped - p2) / (p98 - p2) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(data, dtype=np.uint8)
    return normalized


def cut_patches(
    safe_dir: Path,
    source_tile_id: str,
    patch_size: int | None = None,
    overlap_ratio: float | None = None,
    min_coverage: float | None = None,
    gsd_m: float | None = None,
    aoi_bbox: list[float] | tuple[float, float, float, float] | None = None,
) -> Generator[PatchMeta, None, None]:
    """
    Нарезать .SAFE → патчи PNG в MinIO.

    Yields PatchMeta для каждого сохранённого патча.
    """
    patch_size = patch_size or _s.patch_size_px
    overlap_ratio = overlap_ratio or _s.patch_overlap_ratio
    min_coverage = min_coverage or _s.patch_min_coverage
    gsd_m = gsd_m or _s.patch_gsd_m

    step = int(patch_size * (1 - overlap_ratio))

    band_paths = find_band_files(safe_dir)

    with (
        rasterio.open(band_paths["R"]) as r_src,
        rasterio.open(band_paths["G"]) as g_src,
        rasterio.open(band_paths["B"]) as b_src,
    ):
        width = r_src.width
        height = r_src.height
        transform = r_src.transform
        crs = r_src.crs

        # Перепроецируем трансформацию в EPSG:4326 для координат
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        n_patches = 0
        skipped = 0

        for row_off in range(0, height - patch_size + 1, step):
            for col_off in range(0, width - patch_size + 1, step):
                window = Window(col_off, row_off, patch_size, patch_size)

                # Читаем три канала
                r = r_src.read(1, window=window)
                g = g_src.read(1, window=window)
                b = b_src.read(1, window=window)

                # Проверка покрытия (nodata = 0 в Sentinel-2)
                coverage = np.count_nonzero(r) / r.size
                if coverage < min_coverage:
                    skipped += 1
                    continue

                # Нормализуем и собираем RGB
                rgb = np.stack([
                    _normalize_band(r),
                    _normalize_band(g),
                    _normalize_band(b),
                ], axis=-1)

                # Геопривязка патча
                win_transform = rasterio.windows.transform(window, transform)
                lon_min_p, lat_max_p = to_wgs84.transform(
                    win_transform.c, win_transform.f
                )
                lon_max_p, lat_min_p = to_wgs84.transform(
                    win_transform.c + patch_size * win_transform.a,
                    win_transform.f + patch_size * win_transform.e,
                )
                center_lon = (lon_min_p + lon_max_p) / 2
                center_lat = (lat_min_p + lat_max_p) / 2

                if aoi_bbox is not None:
                    aoi_lon_min, aoi_lat_min, aoi_lon_max, aoi_lat_max = aoi_bbox
                    intersects_aoi = not (
                        lon_max_p < aoi_lon_min
                        or lon_min_p > aoi_lon_max
                        or lat_max_p < aoi_lat_min
                        or lat_min_p > aoi_lat_max
                    )
                    if not intersects_aoi:
                        skipped += 1
                        continue

                # Сохранить в MinIO
                patch_id = str(uuid.uuid4())
                s3_key = f"patches/{source_tile_id}/{patch_id}.png"

                img = Image.fromarray(rgb)
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=False)
                png_bytes = buf.getvalue()

                upload_bytes(png_bytes, s3_key, content_type="image/png")

                n_patches += 1
                yield PatchMeta(
                    patch_id=patch_id,
                    s3_key=s3_key,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    bbox=(lon_min_p, lat_min_p, lon_max_p, lat_max_p),
                    patch_size=patch_size,
                    gsd_m=gsd_m,
                )

        logger.info(
            "tile_cut_done",
            safe_dir=str(safe_dir.name),
            patches=n_patches,
            skipped=skipped,
        )


def cut_patches_from_raster(
    raster_path: Path,
    source_tile_id: str,
    patch_size: int | None = None,
    overlap_ratio: float | None = None,
    min_coverage: float | None = None,
    gsd_m: float | None = None,
    aoi_bbox: list[float] | tuple[float, float, float, float] | None = None,
    normalize: bool = False,
) -> Generator[PatchMeta, None, None]:
    """
    Обобщённая нарезка ЛЮБОГО 3-канального георефернс-растра (например, мозаики
    Esri World Imagery из `esri_client.fetch_mosaic_to_geotiff`) на патчи PNG в
    MinIO. В отличие от `cut_patches` (заточена под трёхфайловый Sentinel .SAFE),
    читает RGB из одного многоканального датасета.

    normalize=False по умолчанию: снимки Esri уже 8-битные и радиометрически
    сбалансированные, percentile-стретч не нужен (в отличие от 16-битных
    Sentinel-каналов). Согласование доменов UAV↔эталон делается позже единой
    предобработкой перед SIFT (см. services/features/preprocess.py), а не здесь.

    Yields PatchMeta для каждого сохранённого патча.
    """
    patch_size = patch_size or _s.patch_size_px
    overlap_ratio = _s.patch_overlap_ratio if overlap_ratio is None else overlap_ratio
    min_coverage = _s.patch_min_coverage if min_coverage is None else min_coverage
    gsd_m = gsd_m or _s.patch_gsd_m

    step = int(patch_size * (1 - overlap_ratio))

    with rasterio.open(raster_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs

        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        n_patches = 0
        skipped = 0

        for row_off in range(0, height - patch_size + 1, step):
            for col_off in range(0, width - patch_size + 1, step):
                window = Window(col_off, row_off, patch_size, patch_size)
                bands = src.read(indexes=[1, 2, 3], window=window)  # (3, h, w)
                rgb = np.transpose(bands, (1, 2, 0)).astype(np.uint8)

                # Покрытие: доля непустых (не nodata=0) пикселей по любому каналу
                valid = np.any(rgb > 0, axis=2)
                coverage = float(valid.mean())
                if coverage < min_coverage:
                    skipped += 1
                    continue

                if normalize:
                    rgb = np.stack(
                        [_normalize_band(rgb[..., i]) for i in range(3)], axis=-1
                    )

                # Геопривязка патча (CRS растра → WGS84)
                win_transform = rasterio.windows.transform(window, transform)
                lon_min_p, lat_max_p = to_wgs84.transform(
                    win_transform.c, win_transform.f
                )
                lon_max_p, lat_min_p = to_wgs84.transform(
                    win_transform.c + patch_size * win_transform.a,
                    win_transform.f + patch_size * win_transform.e,
                )
                center_lon = (lon_min_p + lon_max_p) / 2
                center_lat = (lat_min_p + lat_max_p) / 2

                if aoi_bbox is not None:
                    aoi_lon_min, aoi_lat_min, aoi_lon_max, aoi_lat_max = aoi_bbox
                    intersects_aoi = not (
                        lon_max_p < aoi_lon_min
                        or lon_min_p > aoi_lon_max
                        or lat_max_p < aoi_lat_min
                        or lat_min_p > aoi_lat_max
                    )
                    if not intersects_aoi:
                        skipped += 1
                        continue

                patch_id = str(uuid.uuid4())
                s3_key = f"patches/{source_tile_id}/{patch_id}.png"

                img = Image.fromarray(rgb)
                import io
                buf = io.BytesIO()
                img.save(buf, format="PNG", optimize=False)
                upload_bytes(buf.getvalue(), s3_key, content_type="image/png")

                n_patches += 1
                yield PatchMeta(
                    patch_id=patch_id,
                    s3_key=s3_key,
                    center_lon=center_lon,
                    center_lat=center_lat,
                    bbox=(lon_min_p, lat_min_p, lon_max_p, lat_max_p),
                    patch_size=patch_size,
                    gsd_m=gsd_m,
                )

        logger.info(
            "raster_cut_done",
            raster=str(Path(raster_path).name),
            patches=n_patches,
            skipped=skipped,
        )
