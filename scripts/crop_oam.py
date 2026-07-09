#!/usr/bin/env python
"""Crop a clean square window from a (remote) georeferenced COG.

Reads a lon/lat window directly from the Cloud-Optimized GeoTIFF via /vsicurl/,
verifies the window is fully inside valid (non-transparent) imagery, and writes
a plain RGB square JPEG resembling a single drone capture. Prints the exact
ground-truth center/bbox so the localization result can be checked objectively.
"""
from __future__ import annotations

import sys

import click
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds


@click.command()
@click.option("--url", required=True, help="COG URL (https, COG-range-read capable)")
@click.option("--center-lon", type=float, required=True)
@click.option("--center-lat", type=float, required=True)
@click.option("--size-m", type=float, default=1200.0, show_default=True, help="Square side in meters")
@click.option("--out", required=True, type=click.Path(), help="Output JPEG path")
@click.option("--max-px", type=int, default=2000, show_default=True, help="Max output side in pixels")
def main(url, center_lon, center_lat, size_m, out, max_px):
    half_lat = (size_m / 2.0) / 111320.0
    half_lon = (size_m / 2.0) / (111320.0 * np.cos(np.radians(center_lat)))
    lon_min, lon_max = center_lon - half_lon, center_lon + half_lon
    lat_min, lat_max = center_lat - half_lat, center_lat + half_lat

    vsi = f"/vsicurl/{url}"
    with rasterio.Env(GDAL_HTTP_MULTIPLEX="YES", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
        with rasterio.open(vsi) as src:
            win = from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
            # Decide an output resolution capped at max_px
            win_w, win_h = win.width, win.height
            scale = min(1.0, max_px / max(win_w, win_h))
            out_w, out_h = max(1, round(win_w * scale)), max(1, round(win_h * scale))

            bands = src.read(
                indexes=[1, 2, 3],
                window=win,
                out_shape=(3, out_h, out_w),
                resampling=Resampling.bilinear,
                boundless=True,
                fill_value=0,
            )
            # Validity mask (alpha band if present, else non-black)
            if src.count >= 4:
                alpha = src.read(
                    4, window=win, out_shape=(out_h, out_w),
                    resampling=Resampling.nearest, boundless=True, fill_value=0,
                )
                valid = alpha > 0
            else:
                valid = np.any(bands > 0, axis=0)

    valid_frac = float(valid.mean())
    rgb = np.transpose(bands, (1, 2, 0))  # H,W,3 (RGB)

    click.echo(f"center=({center_lon:.6f},{center_lat:.6f}) size_m={size_m}")
    click.echo(f"bbox=[{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}]")
    click.echo(f"out_px={out_w}x{out_h} valid_fraction={valid_frac:.3f}")
    if valid_frac < 0.999:
        click.secho("WARNING: window contains transparent/nodata pixels — shift center or shrink.", fg="yellow")

    import cv2
    bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(out, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    click.secho(f"wrote {out}", fg="green")


if __name__ == "__main__":
    main()
