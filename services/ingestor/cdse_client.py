"""
Клиент для Copernicus Data Space Ecosystem (CDSE).

Документация: https://documentation.dataspace.copernicus.eu/APIs/OData.html
Аутентификация: OAuth2 password grant через Keycloak (логин + пароль от аккаунта CDSE).
Создавать отдельный OAuth client не нужно — используется публичный client_id "cdse-public".
"""
import time
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_logger, get_settings

logger = get_logger(__name__)
_settings = get_settings()

_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu"
    "/auth/realms/CDSE/protocol/openid-connect/token"
)
_CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu"

# Публичный client_id CDSE — не нужно создавать отдельный OAuth client
_CDSE_PUBLIC_CLIENT_ID = "cdse-public"


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=20))
def _catalog_get(url: str) -> requests.Response:
    return requests.get(url, timeout=60)


class CDSEToken:
    """
    OAuth2 token через password grant.
    Использует логин/пароль от аккаунта dataspace.copernicus.eu.
    """

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: float = 0.0
        self._refresh_expires_at: float = 0.0

    def get_access_token(self) -> str:
        if time.time() < self._expires_at - 30:
            return self._access_token  # type: ignore[return-value]
        # Попробовать обновить через refresh_token
        if self._refresh_token and time.time() < self._refresh_expires_at - 30:
            self._refresh()
        else:
            self._fetch_token()
        return self._access_token  # type: ignore[return-value]

    def _fetch_token(self) -> None:
        """Получить новый токен через username/password."""
        data = {
            "grant_type": "password",
            "client_id": _CDSE_PUBLIC_CLIENT_ID,
            "username": self._username,
            "password": self._password,
        }
        resp = requests.post(_TOKEN_URL, data=data, timeout=30)
        resp.raise_for_status()
        self._save(resp.json())

    def _refresh(self) -> None:
        """Обновить токен через refresh_token."""
        data = {
            "grant_type": "refresh_token",
            "client_id": _CDSE_PUBLIC_CLIENT_ID,
            "refresh_token": self._refresh_token,
        }
        resp = requests.post(_TOKEN_URL, data=data, timeout=30)
        if resp.status_code != 200:
            self._fetch_token()
            return
        self._save(resp.json())

    def _save(self, payload: dict) -> None:
        self._access_token = payload["access_token"]
        self._refresh_token = payload.get("refresh_token")
        self._expires_at = time.time() + payload.get("expires_in", 600)
        self._refresh_expires_at = time.time() + payload.get("refresh_expires_in", 3600)


class CDSEClient:
    """
    Клиент для поиска и скачивания продуктов Sentinel-2 из CDSE.

    Аутентификация: логин и пароль от аккаунта dataspace.copernicus.eu
    (переменные CDSE_USERNAME и CDSE_PASSWORD в .env).

    Пример использования:
        client = CDSEClient()
        products = client.search(
            bbox=[35.0, 50.0, 40.0, 55.0],
            date_from=date(2023, 6, 1),
            date_to=date(2024, 9, 30),
            cloud_cover_max=20,
        )
        for p in products:
            path = client.download(p["Id"], output_dir=Path("./downloads"))
    """

    def __init__(self) -> None:
        self._token = CDSEToken(
            username=_settings.cdse_username,
            password=_settings.cdse_password,
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        bbox: list[float],
        date_from: date,
        date_to: date,
        cloud_cover_max: float = 20.0,
        collection: str = "SENTINEL-2",
        product_type: str = "S2MSI2A",
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Поиск продуктов по bbox + временному диапазону + облачности.

        bbox: [lon_min, lat_min, lon_max, lat_max]
        Возвращает список словарей с полями Id, Name, ContentDate, ...
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        aoi_wkt = (
            f"POLYGON(({lon_min} {lat_min},"
            f"{lon_max} {lat_min},"
            f"{lon_max} {lat_max},"
            f"{lon_min} {lat_max},"
            f"{lon_min} {lat_min}))"
        )

        filters = [
            f"Collection/Name eq '{collection}'",
            f"Attributes/OData.CSC.StringAttribute/any("
            f"att:att/Name eq 'productType' and att/OData.CSC.StringAttribute/Value eq '{product_type}')",
            f"Attributes/OData.CSC.DoubleAttribute/any("
            f"att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {cloud_cover_max})",
            f"OData.CSC.Intersects(area=geography'SRID=4326;{aoi_wkt}')",
            f"ContentDate/Start gt {date_from.isoformat()}T00:00:00.000Z",
            f"ContentDate/Start lt {date_to.isoformat()}T23:59:59.000Z",
        ]
        query = " and ".join(filters)

        results = []
        skip = 0
        page_size = min(max_results, 100)

        while len(results) < max_results:
            url = (
                f"{_CATALOG_URL}"
                f"?$filter={query}"
                f"&$top={page_size}"
                f"&$skip={skip}"
                f"&$orderby=ContentDate/Start asc"
            )
            resp = _catalog_get(url)
            resp.raise_for_status()
            page = resp.json().get("value", [])
            if not page:
                break
            results.extend(page)
            skip += len(page)
            if len(page) < page_size:
                break

        logger.info("cdse_search_done", count=len(results), bbox=bbox)
        return results[:max_results]

    # ── Download ──────────────────────────────────────────────────────────────

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=5, max=30))
    def download(
        self,
        product_id: str,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> Path:
        """
        Скачать продукт по UUID в output_dir.
        Возвращает путь к скачанному zip-файлу.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        token = self._token.get_access_token()

        # Получаем имя файла через OData
        meta_url = f"{_CATALOG_URL}({product_id})"
        meta_resp = requests.get(
            meta_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
        )
        meta_resp.raise_for_status()
        product_name = meta_resp.json().get("Name", product_id)
        out_path = output_dir / f"{product_name}.zip"

        if skip_existing and out_path.exists():
            logger.info("cdse_skip_existing", path=str(out_path))
            return out_path

        download_url = f"{_DOWNLOAD_BASE}/odata/v1/Products({product_id})/$value"
        logger.info("cdse_download_start", product_id=product_id, url=download_url)

        with httpx.stream(
            "GET",
            download_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=None,
            follow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(out_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)

        logger.info(
            "cdse_download_done",
            product_id=product_id,
            size_mb=round(downloaded / 1024 / 1024, 1),
            path=str(out_path),
        )
        return out_path
