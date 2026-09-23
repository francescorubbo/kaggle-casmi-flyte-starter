"""Tiny helpers to move pyarrow tables in and out of flyte.io.File.

Two formats, chosen by the file name passed to `write_table`:
* `.parquet` - compact, the default for most intermediate files.
* `.arrow` (Arrow IPC / Feather v2) - for big fingerprint matrices. Parquet stores fixed-size lists
  as repeated fields, and reading them back materialises two int16 "levels" per value: for
  277k molecules x 4860 bits that is several GB on top of the data. IPC stores them as-is.
`read_table` detects the format from the file itself.
"""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.parquet as pq
from flyte.io import File


def read_local_table(path: str, columns: list[str] | None = None) -> pa.Table:
    with open(path, "rb") as fh:
        magic = fh.read(6)
    if magic == b"ARROW1":
        return feather.read_table(path, columns=columns, memory_map=True)
    return pq.read_table(path, columns=columns)


async def read_table(f: File, columns: list[str] | None = None) -> pa.Table:
    return read_local_table(await f.download(), columns=columns)


async def write_table(table: pa.Table, name: str) -> File:
    path = Path(tempfile.mkdtemp()) / name
    if name.endswith(".arrow"):
        feather.write_feather(table, str(path), compression="zstd")
    else:
        pq.write_table(table, path, compression="zstd")
    return await File.from_local(str(path))


def matrix_column(x: np.ndarray) -> pa.FixedSizeListArray:
    """(n, d) numpy array -> fixed-size-list column, the natural parquet layout for fingerprints."""
    x = np.ascontiguousarray(x)
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def column_matrix(col: pa.ChunkedArray | pa.FixedSizeListArray, mask: np.ndarray | None = None) -> np.ndarray:
    """Inverse of matrix_column, optionally keeping only rows where `mask` is True.

    Fills one preallocated array chunk by chunk (zero-copy views of the Arrow buffers), so the
    peak is the Arrow data plus the result - not several full-size temporaries.
    """
    chunks = col.chunks if isinstance(col, pa.ChunkedArray) else [col]
    width = col.type.list_size
    n = int(mask.sum()) if mask is not None else len(col)
    out, pos, start = None, 0, 0
    for chunk in chunks:
        values = chunk.values.slice(chunk.offset * width, len(chunk) * width)
        block = values.to_numpy(zero_copy_only=values.null_count == 0).reshape(-1, width)
        if mask is not None:
            block = block[mask[start : start + len(chunk)]]
        if out is None:
            out = np.empty((n, width), dtype=block.dtype)
        out[pos : pos + len(block)] = block
        pos, start = pos + len(block), start + len(chunk)
    return out if out is not None else np.empty((0, width))


def stable_fraction(keys: pa.Array) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per key (same on every machine and every run)."""
    import hashlib

    return np.array(
        [int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest()) / 2**64 for k in keys.to_pylist()]
    )
