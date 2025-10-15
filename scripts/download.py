#!/usr/bin/env python3 
import argparse
from os import environ
import asyncio
import hashlib
from logging import basicConfig, getLogger
from pathlib import Path
from zipfile import ZipFile

import httpx
from tqdm import tqdm

CHUNK = 8192
logger = getLogger("comfi_downloads")


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Zenodo downloader")
    p.add_argument("zenodo_id", nargs="?", default="17223909")
    p.add_argument(
        "-d",
        "--download-dir",
        type=Path,
        default=Path("downloads"),
    )
    p.add_argument("--delete-zip", action="store_true")
    p.add_argument(
        "--comfi-root",
        type=Path,
        default=Path(environ.get("COMFI_ROOT", "COMFI")),
    )
    p.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=5,
    )
    p.add_argument(
        "-q",
        "--quiet",
        action="count",
        default=int(environ.get("QUIET", 0)),
        help="decrement verbosity level",
    )

    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=int(environ.get("VERBOSITY", 0)),
        help="increment verbosity level",
    )
    return p


class Entry:
    def __init__(self, name: str, meta: dict, download_dir: Path):
        self.name = name
        self.url = meta["links"]["content"]
        self.checksum = meta["checksum"]
        self.size = meta["size"]
        self.download_dir = download_dir
        self.path = download_dir / name
        self.tmp_path = download_dir / f"{name}.part"
        logger.debug(f"created entry {name}")

    def __str__(self):
        return self.name

    async def download(self, client: httpx.AsyncClient):
        if self.path.exists():
            logger.info(f"{self.path} already exist")
            try:
                self.verify_checksum(self.path)
                logger.info("%s has correct checksum, skipping", self.path)
                return
            except ValueError as e:
                logger.warning("removing wrong %s, because: %s", self.name, e)
                self.path.unlink()

        progress = tqdm(desc=self.name, total=self.size, unit="B", unit_scale=True)

        async with client.stream("GET", self.url, follow_redirects=True) as resp:
            resp.raise_for_status()
            with self.tmp_path.open("wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=CHUNK):
                    f.write(chunk)
                    progress.update(len(chunk))

        progress.close()
        self.verify_checksum(self.tmp_path)
        self.tmp_path.rename(self.path)

    def verify_checksum(self, path: Path):
        algo, expected = self.checksum.split(":")
        if not hasattr(hashlib, algo):
            err = f"hashlib does not know {algo}"
            raise NotImplementedError(err)

        h = getattr(hashlib, algo)()
        with path.open("rb") as f:
            while chunk := f.read(CHUNK):
                h.update(chunk)

        digest = h.hexdigest()
        if digest != expected:
            err = f"wrong {algo} checksum for {self.name}: {digest} != {expected}"
            raise ValueError(err)

    def extract(self, delete_zip: bool):
        out = self.download_dir / self.path.stem
        if out.exists():
            logger.info("%s already extracted, skipping", out)
            return

        if self.path.suffix == ".zip":
            with ZipFile(self.path) as z:
                z.extractall(out)
            if delete_zip:
                logger.info("removing %s", self.path)
                self.path.unlink()
        else:
            logger.warning("unknown extension %s", self.path.suffix)


async def fetch_entries(
    client: httpx.AsyncClient, record_id: str, download_dir: Path
) -> list[Entry]:
    url = f"https://zenodo.org/records/{record_id}/export/json"

    r = await client.get(url)
    r.raise_for_status()
    data = r.json()["files"]["entries"]

    return [Entry(name, meta, download_dir) for name, meta in data.items()]


async def main(
    zenodo_id: str,
    download_dir: Path,
    comfi_root: Path,
    jobs: int,
    delete_zip: bool,
    **kwargs,
):
    download_dir.mkdir(parents=True, exist_ok=True)
    limits = httpx.Limits(max_connections=jobs)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        logger.info("Requesting files from zenodo…")
        entries = await fetch_entries(client, zenodo_id, download_dir)

        logger.info("Downloading entries")
        await asyncio.gather(*(entry.download(client) for entry in entries))

        logger.info("Extracting entries")
        for entry in entries:
            entry.extract(delete_zip)

    logger.info("Generating %s directory", comfi_root)

    # Symlink videos from download_dir to comfi_root
    path = comfi_root / "videos"
    path.mkdir(parents=True, exist_ok=True)
    for folder in download_dir.glob("videos*"):
        if not folder.is_dir():
            continue
        for child in (folder / folder.name).iterdir():
            (path / child.name).symlink_to(
                target=child.absolute(), target_is_directory=child.is_dir()
            )

    # Symling everything else
    for folder in ["cam_params", "forces", "mocap", "robot", "metadata"]:
        path = comfi_root / folder
        path.mkdir(parents=True, exist_ok=True)
        for child in (download_dir / folder / folder).iterdir():
            (path / child.name).symlink_to(
                target=child.absolute(), target_is_directory=child.is_dir()
            )


if __name__ == "__main__":
    args = get_parser().parse_args()
    basicConfig(level=30 - 10 * args.verbose + 10 * args.quiet)
    asyncio.run(main(**vars(args)))
