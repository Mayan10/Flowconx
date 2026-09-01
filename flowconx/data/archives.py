"""Streaming readers for the compressed raw archives.

The raw corpora are 3.2 GB (5G Traffic) and 21 GB (CESNET-QUIC22) as zip
files, and expanding them is not an option on a machine with 26 GB free. Every
loader in this package therefore reads members directly out of the archive.

Two consequences worth stating, because they shape the loaders:

* Zip members are read sequentially. There is no random access inside a
  member, so every pass over the data is a single forward scan and anything
  that needs sampling has to be a reservoir, not a shuffle-then-take.
* A ``.csv.gz`` inside a ``.zip`` is doubly compressed. ``gzip.open`` over the
  raw member stream handles it, at the cost of both codecs running on every
  byte, so throughput is the binding constraint on CESNET rather than disk.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class ArchiveMember:
    """One file inside an archive, with the sizes needed for progress reporting."""

    archive: Path
    name: str
    size: int
    compressed_size: int

    @property
    def stem(self) -> str:
        return self.name.rsplit("/", 1)[-1]

    @property
    def parts(self) -> List[str]:
        return self.name.split("/")


def sha256_of_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    """SHA256 of an archive, for the checksum table in the data documentation."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_members(archive: str | Path, suffixes: Sequence[str] = (".csv",)) -> List[ArchiveMember]:
    """Every member whose name ends in one of ``suffixes``, in sorted order.

    Sorted so that the set of files processed under a row budget is a
    deterministic function of the budget, not of zip directory order.
    """
    path = Path(archive)
    with zipfile.ZipFile(path) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
    selected = [info for info in infos if any(info.filename.endswith(s) for s in suffixes)]
    selected.sort(key=lambda info: info.filename)
    return [
        ArchiveMember(archive=path, name=info.filename, size=info.file_size, compressed_size=info.compress_size)
        for info in selected
    ]


class text_stream:  # noqa: N801 - used as a context manager, reads as one
    """Context manager yielding a decoded text stream for an archive member.

        with text_stream(member) as handle:
            for line in handle:
                ...
    """

    def __init__(
        self,
        member: ArchiveMember,
        encoding: str = "utf-8-sig",
        errors: str = "replace",
        gunzip: Optional[bool] = None,
    ) -> None:
        self.member = member
        self.encoding = encoding
        self.errors = errors
        self.gunzip = member.name.endswith(".gz") if gunzip is None else gunzip
        self._zip: Optional[zipfile.ZipFile] = None
        self._handles: List[io.IOBase] = []

    def __enter__(self) -> io.TextIOBase:
        self._zip = zipfile.ZipFile(self.member.archive)
        raw = self._zip.open(self.member.name)
        self._handles.append(raw)
        if self.gunzip:
            raw = gzip.open(raw, "rb")
            self._handles.append(raw)
        # A large buffer matters here: the outer zip and inner gzip codecs both
        # run per read, so small reads dominate the cost on the 21 GB archive.
        buffered = io.BufferedReader(raw, buffer_size=1 << 22)
        self._handles.append(buffered)
        text = io.TextIOWrapper(buffered, encoding=self.encoding, errors=self.errors)
        self._handles.append(text)
        return text

    def __exit__(self, *exc_info: object) -> None:
        for handle in reversed(self._handles):
            try:
                handle.close()
            except (OSError, ValueError):
                pass
        self._handles.clear()
        if self._zip is not None:
            self._zip.close()
            self._zip = None
