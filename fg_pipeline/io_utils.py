from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, TypeVar

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is an optional UX enhancement at runtime.
    tqdm = None

_T = TypeVar("_T")


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    path = ensure_parent_dir(path)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False))
            handle.write("\n")


def count_jsonl_rows(path: str | Path) -> int:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def maybe_tqdm(
    iterable: Iterable[_T],
    *,
    desc: str,
    total: int | None = None,
) -> Iterable[_T]:
    if tqdm is None:
        return iterable
    return tqdm(iterable, desc=desc, total=total, dynamic_ncols=True)
