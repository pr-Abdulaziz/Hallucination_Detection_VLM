#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path


VG_ARCHIVES = {
    "images.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "images2.zip": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}
CHUNK_SIZE = 1024 * 1024


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def download_file(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination.name} already exists")
        return

    print(f"[download] {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as output:
            total_bytes = int(response.headers.get("Content-Length", "0"))
            downloaded = 0

            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total_bytes:
                    percent = downloaded * 100 / total_bytes
                    print(
                        f"\r  -> {downloaded / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r  -> {downloaded / 1024**3:.2f} GB",
                        end="",
                        flush=True,
                    )
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    print()


def extract_flat(zip_path: Path, image_dir: Path) -> None:
    print(f"[extract] {zip_path.name} -> {image_dir}")
    extracted = 0
    skipped = 0

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            filename = Path(member.filename).name
            if not filename:
                continue

            target_path = image_dir / filename
            if target_path.exists():
                skipped += 1
                continue

            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target, CHUNK_SIZE)
            extracted += 1

    print(f"  -> extracted {extracted} files, skipped {skipped} existing files")


def main() -> int:
    repo_root = get_repo_root()
    vg_root = repo_root / "vg"
    image_dir = vg_root / "images"
    download_dir = vg_root / "_downloads"

    image_dir.mkdir(parents=True, exist_ok=True)
    download_dir.mkdir(parents=True, exist_ok=True)

    print(f"Repo root: {repo_root}")
    print(f"Visual Genome images will be stored in: {image_dir}")

    for archive_name, url in VG_ARCHIVES.items():
        archive_path = download_dir / archive_name
        download_file(url, archive_path)
        extract_flat(archive_path, image_dir)
        archive_path.unlink(missing_ok=True)
        print(f"[cleanup] removed {archive_path}")

    total_images = len(list(image_dir.glob("*.jpg")))
    print(f"[done] Visual Genome install complete. Found {total_images} jpg files in {image_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
