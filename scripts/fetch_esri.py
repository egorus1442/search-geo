#!/usr/bin/env python
"""Fetch a georeferenced window from Esri World Imagery (ArcGIS REST export).

Analogue of crop_oam.py, but the source is the global Esri World Imagery
basemap (ArcGIS MapServer `export` endpoint) instead of an OpenAerialMap COG.

Requests a bbox (built from center+size in Web-Mercator meters) at a chosen
target GSD, saves a georeferenced GeoTIFF (+ a plain JPEG for inspection),
and prints the exact center/bbox and effective GSD so the localization result
can be checked objectively.

NOTE on licensing: this is intended for *evaluation* pulls (single small AOI,
not published/redistributed). It is NOT a production ingestion path — Esri
basemap terms do not grant rights to build a permanent derivative tile DB.

Example:
    python scripts/fetch_esri.py \
        --center-lon 61.5172 --center-lat 56.2052 \
        --size-m 640 --gsd 0.6 \
        --out /app/esri_bagaryak_gt.tif --out-jpg /app/esri_bagaryak_gt.jpg
"""
from __future__ import annotations

import io
import math

import click
import numpy as np
import requests

_EXPORT_URL = (
    "https://services.arcgisonline.com/arcgis/rest/services/"
    "World_Imagery/MapServer/export"
)
# ArcGIS MapServer usually caps a single export request at 4096 px per side.
_MAX_REQ_PX = 4096


def _lonlat_to_3857(lon: float, lat: float) -> tuple[float, float]:
    r = 6378137.0
    x = r * math.radians(lon)
    y = r * math.log(math.tan(math.pi / 4.0 + math.radians(lat) / 2.0))
    return x, y


def _export_window(
    xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int
) -> np.ndarray:
    """One ArcGIS export request → HxWx3 uint8 RGB array (Web-Mercator bbox)."""
    params = {
        "bbox": f"{xmin},{ymin},{xmax},{ymax}",
        "bboxSR": 3857,
        "imageSR": 3857,
        "size": f"{w},{h}",
        "format": "tiff",
        "f": "image",
    }
    resp = requests.get(_EXPORT_URL, params=params, timeout=120)
    resp.raise_for_status()
    import rasterio

    with rasterio.open(io.BytesIO(resp.content)) as src:
        bands = src.read(indexes=[1, 2, 3])  # RGB
    return np.transpose(bands, (1, 2, 0)).astype(np.uint8)


@click.command()
@click.option("--center-lon", type=float, required=True)
@click.option("--center-lat", type=float, required=True)
@click.option("--size-m", type=float, default=640.0, show_default=True, help="Square side in meters")
@click.option("--gsd", type=float, default=0.6, show_default=True, help="Target ground sample distance, m/px")
@click.option("--out", required=True, type=click.Path(), help="Output GeoTIFF path")
@click.option("--out-jpg", default=None, type=click.Path(), help="Optional plain JPEG for inspection")
def main(center_lon, center_lat, size_m, gsd, out, out_jpg):
    import rasterio
    from rasterio.transform import from_bounds

    cx, cy = _lonlat_to_3857(center_lon, center_lat)
    half = size_m / 2.0
    xmin, xmax = cx - half, cx + half
    ymin, ymax = cy - half, cy + half

    total_px = int(round(size_m / gsd))
    if total_px <= _MAX_REQ_PX:
        rgb = _export_window(xmin, ymin, xmax, ymax, total_px, total_px)
    else:
        # Tile the request grid so we never exceed per-request / RAM limits.
        n = math.ceil(total_px / _MAX_REQ_PX)
        step_m = (xmax - xmin) / n
        px_per = math.ceil(total_px / n)
        rows = []
        for r in range(n):  # top→bottom (y descending)
            cols = []
            for c in range(n):
                wx0 = xmin + c * step_m
                wx1 = wx0 + step_m
                wy1 = ymax - r * step_m
                wy0 = wy1 - step_m
                cols.append(_export_window(wx0, wy0, wx1, wy1, px_per, px_per))
            rows.append(np.hstack(cols))
        rgb = np.vstack(rows)
        total_px = rgb.shape[0]

    h, w = rgb.shape[:2]
    valid = np.any(rgb > 0, axis=2)
    valid_frac = float(valid.mean())

    transform = from_bounds(xmin, ymin, xmax, ymax, w, h)
    with rasterio.open(
        out, "w", driver="GTiff", height=h, width=w, count=3,
        dtype="uint8", crs="EPSG:3857", transform=transform,
    ) as dst:
        for i in range(3):
            dst.write(rgb[:, :, i], i + 1)

    click.echo(f"center=({center_lon:.6f},{center_lat:.6f}) size_m={size_m} gsd={gsd}")
    click.echo(f"bbox_3857=[{xmin:.2f},{ymin:.2f},{xmax:.2f},{ymax:.2f}]")
    click.echo(f"out_px={w}x{h} valid_fraction={valid_frac:.3f}")
    click.echo(f"wrote {out}")

    if out_jpg:
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_jpg, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        click.echo(f"wrote {out_jpg}")


if __name__ == "__main__":
    main()
