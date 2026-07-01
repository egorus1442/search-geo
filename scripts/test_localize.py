#!/usr/bin/env python
"""
Быстрый тест локализации через HTTP API.

Пример:
    python scripts/test_localize.py --image /path/to/drone_photo.jpg
    python scripts/test_localize.py --image photo.jpg --api-url http://localhost:8000 --top-n 5
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import requests


@click.command()
@click.option("--image", required=True, type=click.Path(exists=True))
@click.option("--api-url", default="http://localhost:8000", show_default=True)
@click.option("--top-n", default=10, show_default=True)
def main(image, api_url, top_n):
    img_path = Path(image)
    click.echo(f"Sending {img_path.name} ({img_path.stat().st_size // 1024} KB) to {api_url}...")

    with open(img_path, "rb") as f:
        resp = requests.post(
            f"{api_url}/api/v1/localize",
            files={"image": (img_path.name, f, "image/jpeg")},
            data={"top_n": top_n},
            timeout=120,
        )

    resp.raise_for_status()
    data = resp.json()

    click.echo(f"\nStatus: {data['status']}")
    click.echo(f"Processing time: {data.get('processing_time_ms', '?')} ms")
    click.echo(f"Candidates: {len(data['candidates'])}")
    click.echo("")

    for c in data["candidates"]:
        click.echo(
            f"  #{c['rank']} patch_id={c['patch_id']} "
            f"lat={c['center_lat']:.4f} lon={c['center_lon']:.4f} "
            f"inliers={c['inlier_count']} conf={c['confidence']:.2f}"
        )


if __name__ == "__main__":
    main()
