"""
Источник эталонной базы: Esri World Imagery (ArcGIS REST `export`).

Заменяет Sentinel-2/CDSE. Тянет георефернс-мозаику World Imagery на заданный
bbox с целевым GSD и пишет её в GeoTIFF (EPSG:3857) ПОБЛОЧНО — так вся мозаика
никогда не держится в RAM целиком (пик памяти ~размер одного блока), а
tile_cutter затем режет GeoTIFF на патчи так же, как раньше резал Sentinel.

Почему Esri: на Sentinel-2 (10 м/пкс) однородная местность (поля, луга,
грунтовки) не даёт повторяемых точек — verifier (SIFT+RANSAC) слепнет. На
~0.5–1 м появляются кромки полей, развилки дорог, контуры, застройка — снова
есть за что цепляться. См. историю выбора источника.

ВНИМАНИЕ по лицензии: предназначено для ОЦЕНОЧНЫХ пулов (небольшой AOI, без
перепубликации). Условия Esri basemap не дают прав на построение постоянной
производной базы тайлов для коммерческого продукта.
"""
from __future__ import annotations

import io
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.transform import from_bounds
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_logger, get_settings

logger = get_logger(__name__)
_s = get_settings()


def _lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    """WGS84 (градусы) → Web-Mercator EPSG:3857 (метры)."""
    r = 6378137.0
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


@retry(stop=stop_after_attempt(4), wait=wait_exponential(min=2, max=20))
def _export_block(
    xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int, url: str
) -> np.ndarray:
    """Один ArcGIS export-запрос → RGB-массив (3, h, w) uint8 в bbox (3857)."""
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{w},{h}",
        "format": "tiff",
        "f": "image",
    }
    resp = requests.get(url, params=params, timeout=180)
    resp.raise_for_status()
    # f=image отдаёт растр без встроенного георефа — берём только пиксели,
    # геопривязку задаём сами по запрошенному bbox (см. fetch_mosaic_to_geotiff).
    with rasterio.open(io.BytesIO(resp.content)) as src:
        return src.read(indexes=[1, 2, 3]).astype(np.uint8)


def fetch_mosaic_to_geotiff(
    bbox: list[float] | tuple[float, float, float, float],
    out_path: str | Path,
    gsd_m: float | None = None,
    max_req_px: int | None = None,
    url: str | None = None,
    workers: int | None = None,
) -> tuple[Path, tuple[int, int]]:
    """
    Скачать мозаику Esri World Imagery на bbox и записать в GeoTIFF (EPSG:3857).

    bbox: [lon_min, lat_min, lon_max, lat_max] (WGS84).
    Возвращает (путь к GeoTIFF, (width_px, height_px)).

    Блоки запрашиваются ПАРАЛЛЕЛЬНО (ThreadPool): рендер export у Esri медленный
    (~10 с на блок), но это сетевой I/O — потоки дают почти линейное ускорение.
    Запись в GeoTIFF идёт из ГЛАВНОГО потока по мере готовности блоков (rasterio
    dst не потокобезопасен на запись). Мировые границы блока берутся из
    трансформации растра (`window_bounds`) — пиксели точно ложатся в сетку.

    NB: размер блока ≤ ~1024 px — у публичного export-эндпоинта World Imagery
    большие запросы (2048+) зависают/отбиваются (см. историю подбора).
    """
    gsd_m = gsd_m or _s.esri_gsd_m
    max_req_px = max_req_px or _s.esri_max_request_px
    url = url or _s.esri_export_url
    workers = workers or _s.esri_fetch_workers

    lon_min, lat_min, lon_max, lat_max = bbox
    xmin, ymin = _lonlat_to_3857(lon_min, lat_min)
    xmax, ymax = _lonlat_to_3857(lon_max, lat_max)

    total_w = max(1, int(round((xmax - xmin) / gsd_m)))
    total_h = max(1, int(round((ymax - ymin) / gsd_m)))
    transform = from_bounds(xmin, ymin, xmax, ymax, total_w, total_h)

    out_path = Path(out_path)
    n_cols = math.ceil(total_w / max_req_px)
    n_rows = math.ceil(total_h / max_req_px)
    n_blocks = n_rows * n_cols

    logger.info(
        "esri_mosaic_start",
        bbox=list(bbox), gsd=gsd_m, width=total_w, height=total_h,
        blocks=n_blocks, workers=workers,
    )

    # Список окон-заданий
    windows: list[Window] = []
    for r in range(n_rows):
        row_off = r * max_req_px
        bh = min(max_req_px, total_h - row_off)
        for c in range(n_cols):
            col_off = c * max_req_px
            bw = min(max_req_px, total_w - col_off)
            windows.append(Window(col_off, row_off, bw, bh))

    def _fetch(win: Window) -> np.ndarray:
        left, bottom, right, top = window_bounds(win, transform)
        return _export_block(left, bottom, right, top, int(win.width), int(win.height), url)

    with rasterio.open(
        out_path, "w", driver="GTiff",
        height=total_h, width=total_w, count=3, dtype="uint8",
        crs="EPSG:3857", transform=transform,
        tiled=True, compress="DEFLATE",
    ) as dst:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_fetch, w): w for w in windows}
            for fut in as_completed(futs):
                win = futs[fut]
                dst.write(fut.result(), window=win)  # запись из главного потока
                done += 1
                if done % 10 == 0 or done == n_blocks:
                    logger.info("esri_blocks_progress", done=done, total=n_blocks)

    logger.info("esri_mosaic_done", path=str(out_path), width=total_w, height=total_h)
    return out_path, (total_w, total_h)
