"""
Проверяет что ссылка на скачивание тайла рабочая —
делает HEAD-запрос (без скачивания) и показывает размер файла.

Запуск:
    python scripts/test_download.py           # только проверка ссылки
    python scripts/test_download.py --go      # реально скачать (~600 MB)
"""
import sys
import json
import urllib.parse
import urllib.request
import urllib.error

USERNAME = "egorkarmishen@gmail.com"
PASSWORD = "132132880088.Ru"

TOKEN_URL    = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL  = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu"

BBOX      = [35.9, 49.9, 36.5, 50.3]
DATE_FROM = "2024-08-01"
DATE_TO   = "2024-09-01"


def get_token():
    body = urllib.parse.urlencode({
        "grant_type": "password",
        "client_id":  "cdse-public",
        "username":   USERNAME,
        "password":   PASSWORD,
    }).encode()
    req = urllib.request.Request(TOKEN_URL, data=body)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())["access_token"]


def search_one():
    lon_min, lat_min, lon_max, lat_max = BBOX
    aoi = f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))"
    filt = " and ".join([
        "Collection/Name eq 'SENTINEL-2'",
        "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq 'S2MSI2A')",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi}')",
        f"ContentDate/Start gt {DATE_FROM}T00:00:00.000Z",
        f"ContentDate/Start lt {DATE_TO}T23:59:59.000Z",
    ])
    params = urllib.parse.urlencode({"$filter": filt, "$top": "1"})
    url = f"{CATALOG_URL}?{params}"
    with urllib.request.urlopen(url, timeout=30) as r:
        products = json.loads(r.read()).get("value", [])
    return products[0] if products else None


def check_link(product_id, token):
    """GET с Range: bytes=0-9 — скачивает только первые 10 байт."""
    url = f"{DOWNLOAD_URL}/odata/v1/Products({product_id})/$value"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Range", "bytes=0-9")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            content_range = r.getheader("Content-Range", "")  # bytes 0-9/TOTAL
            content_type  = r.getheader("Content-Type", "?")
            _ = r.read()  # первые 10 байт
            size_mb = 0
            if content_range:
                # "bytes 0-9/629145600" → парсим total
                try:
                    total = int(content_range.split("/")[-1])
                    size_mb = total / 1024 / 1024
                except Exception:
                    pass
            return {"ok": True, "size_mb": size_mb, "content_type": content_type}
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        # 416 = Range not satisfiable, но файл существует
        if e.code == 416:
            return {"ok": True, "size_mb": 0, "content_type": "?", "note": "Range not supported, but file exists"}
        return {"ok": False, "code": e.code, "msg": body[:300]}


def download(product, token):
    pid  = product["Id"]
    name = product.get("Name", pid)
    url  = f"{DOWNLOAD_URL}/odata/v1/Products({pid})/$value"
    out  = f"./test_download/{name}.zip"

    import os, pathlib
    pathlib.Path("./test_download").mkdir(exist_ok=True)

    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    print(f"Downloading to: {out}")
    with urllib.request.urlopen(req) as resp, open(out, "wb") as f:
        total = int(resp.getheader("Content-Length", 0))
        done  = 0
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = f"{done/total*100:.0f}%" if total else "?"
            print(f"\r  {done/1024/1024:6.1f} MB  {pct}", end="", flush=True)
    size = os.path.getsize(out)
    print(f"\nDone: {out}  ({size/1024/1024:.0f} MB)")


def main():
    go = "--go" in sys.argv
    print("=" * 60)
    print("  CDSE Download Check")
    print("=" * 60)

    print("[1] Getting token...", end=" ", flush=True)
    token = get_token()
    print("OK")

    print("[2] Searching for a product...", end=" ", flush=True)
    product = search_one()
    if not product:
        print("Not found")
        sys.exit(1)
    print(f"OK\n    {product['Name']}\n    ID: {product['Id']}")

    print("[3] Checking download link (HEAD request)...", end=" ", flush=True)
    info = check_link(product["Id"], token)
    if info["ok"]:
        print(f"OK")
        print(f"    Size: {info['size_mb']:.0f} MB")
        print(f"    Type: {info['content_type']}")
    else:
        print(f"FAILED: {info['code']} {info['msg']}")
        sys.exit(1)

    if go:
        print(f"\n[4] Starting download ({info['size_mb']:.0f} MB)...")
        download(product, token)
    else:
        print(f"\n    Run with --go to actually download ({info['size_mb']:.0f} MB)")

    print("\nAll checks passed!")


if __name__ == "__main__":
    main()
