#!/usr/bin/env python
"""
Быстрый тест CDSE без сторонних библиотек (только встроенный urllib).
Проверяет: аутентификация → поиск продуктов.

Запуск:
    python scripts/test_cdse.py
    python scripts/test_cdse.py --download   # скачать первый найденный продукт (~600 МБ)
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# ── Конфиг ────────────────────────────────────────────────────────────────────

def load_dotenv() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


load_dotenv()

USERNAME = os.getenv("CDSE_USERNAME", "")
PASSWORD = os.getenv("CDSE_PASSWORD", "")

# Небольшой тестовый bbox — окрестности Харькова
BBOX      = [35.9, 49.9, 36.5, 50.3]   # [lon_min, lat_min, lon_max, lat_max]
DATE_FROM = "2024-05-01"
DATE_TO   = "2024-09-01"
CLOUD_MAX = 30.0

TOKEN_URL    = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL  = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def http_post(url: str, data: dict) -> dict:
    body = urllib.parse.urlencode(data).encode()
    req  = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def http_get(url: str, token: str | None = None) -> dict:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ── Шаг 1: Аутентификация ─────────────────────────────────────────────────────

def get_token() -> str:
    if not USERNAME or not PASSWORD:
        print("ERROR: set CDSE_USERNAME and CDSE_PASSWORD in .env or environment")
        sys.exit(1)

    print("[1] Auth CDSE...", end=" ", flush=True)
    try:
        payload = http_post(TOKEN_URL, {
            "grant_type": "password",
            "client_id":  "cdse-public",
            "username":   USERNAME,
            "password":   PASSWORD,
        })
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"\n  ERROR {e.code}: {body[:300]}")
        sys.exit(1)
    token = payload["access_token"]
    expires = payload.get("expires_in", "?")
    print(f"OK (token valid {expires}s)")
    return token


# ── Шаг 2: Поиск продуктов ────────────────────────────────────────────────────

def search_products(token: str) -> list[dict]:
    lon_min, lat_min, lon_max, lat_max = BBOX
    aoi = (
        f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
        f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    )
    filt = " and ".join([
        "Collection/Name eq 'SENTINEL-2'",
        "Attributes/OData.CSC.StringAttribute/any("
        "att:att/Name eq 'productType' "
        "and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
        f"Attributes/OData.CSC.DoubleAttribute/any("
        f"att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {CLOUD_MAX})",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')",
        f"ContentDate/Start gt {DATE_FROM}T00:00:00.000Z",
        f"ContentDate/Start lt {DATE_TO}T23:59:59.000Z",
    ])
    params = urllib.parse.urlencode({
        "$filter": filt,
        "$top": "5",
        "$orderby": "ContentDate/Start desc",
    })
    url = f"{CATALOG_URL}?{params}"

    print(f"[2] Search Sentinel-2 L2A in bbox {BBOX}...", end=" ", flush=True)
    try:
        data = http_get(url, token)
    except urllib.error.HTTPError as e:
        print(f"\n  ERROR {e.code}: {e.read().decode(errors='replace')[:300]}")
        sys.exit(1)

    products = data.get("value", [])
    print(f"found {len(products)} products")
    return products


# ── Step 3: Print results ─────────────────────────────────────────────────────

def print_products(products: list[dict]) -> None:
    print()
    print("-" * 72)
    print(f"{'#':<3}  {'Date':<12}  {'Cloud':>8}  {'Name'}")
    print("-" * 72)
    for i, p in enumerate(products, 1):
        name  = p.get("Name", "?")
        date_ = p.get("ContentDate", {}).get("Start", "?")[:10]
        cloud = "?"
        for attr in p.get("Attributes", []):
            if attr.get("Name") == "cloudCover":
                cloud = f"{attr['Value']:.1f}%"
        print(f"{i:<3}  {date_:<12}  {cloud:>8}  {name[:46]}")
    print("-" * 72)


# ── Step 4: Download first product (optional) ────────────────────────────────

def download_first(product: dict, token: str) -> None:
    pid  = product["Id"]
    name = product.get("Name", pid)
    out  = Path("./test_download")
    out.mkdir(exist_ok=True)
    path = out / f"{name}.zip"

    url = f"{DOWNLOAD_URL}/odata/v1/Products({pid})/$value"
    print(f"\n[3] Downloading: {name[:55]}")
    print(f"    Output: {path}")

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req) as resp, open(path, "wb") as f:
            total   = int(resp.getheader("Content-Length", 0))
            done    = 0
            chunk   = 4 * 1024 * 1024
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                mb = done / 1024 / 1024
                pct = f"{done/total*100:.0f}%" if total else "?"
                print(f"\r  {mb:6.1f} MB  {pct:>5}", end="", flush=True)
    except urllib.error.HTTPError as e:
        print(f"\n  ERROR {e.code}: {e.read().decode(errors='replace')[:300]}")
        return

    print(f"\n    Done! Size: {path.stat().st_size / 1024 / 1024:.0f} MB")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    do_download = "--download" in sys.argv

    print("=" * 72)
    print("  GeoVision - CDSE Connection Test")
    print("=" * 72)
    print(f"  bbox    : {BBOX}")
    print(f"  period  : {DATE_FROM} -> {DATE_TO}")
    print(f"  clouds  : < {CLOUD_MAX}%")
    print()

    token    = get_token()
    products = search_products(token)

    if not products:
        print("\nNo products found. Try wider bbox or date range.")
        sys.exit(0)

    print_products(products)

    if do_download:
        download_first(products[0], token)
    else:
        print("\n  (add --download flag to download the first product)")

    print("\nTest passed OK!")


if __name__ == "__main__":
    main()
