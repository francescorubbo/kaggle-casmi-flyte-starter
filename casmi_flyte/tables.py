"""Tiny helpers to move pyarrow tables in and out of flyte.io.File (parquet)."""

import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from flyte.io import File


async def read_table(f: File, columns: list[str] | None = None) -> pa.Table:
    return pq.read_table(await f.download(), columns=columns)


async def write_table(table: pa.Table, name: str) -> File:
    path = Path(tempfile.mkdtemp()) / name
    pq.write_table(table, path, compression="zstd")
    return await File.from_local(str(path))


def matrix_column(x: np.ndarray) -> pa.FixedSizeListArray:
    """(n, d) numpy array -> fixed-size-list column, the natural parquet layout for fingerprints."""
    x = np.ascontiguousarray(x)
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def column_matrix(col: pa.ChunkedArray | pa.FixedSizeListArray) -> np.ndarray:
    """Inverse of matrix_column."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    width = col.type.list_size
    return col.flatten().to_numpy(zero_copy_only=False).reshape(-1, width)


def stable_fraction(keys: pa.Array) -> np.ndarray:
    """Deterministic pseudo-random number in [0, 1) per key (same on every machine and every run)."""
    import hashlib

    return np.array(
        [int.from_bytes(hashlib.blake2b(k.encode(), digest_size=8).digest()) / 2**64 for k in keys.to_pylist()]
    )
