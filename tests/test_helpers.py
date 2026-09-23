"""Checks for the provided helpers. `test_bin_spectra_on_real_data` needs a local copy of the data."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from casmi_flyte.baseline import LOSS_BINS, MZ_BINS, bin_spectra
from casmi_flyte.tables import column_matrix, matrix_column

TRAIN = Path(__file__).parents[1] / "data" / "train.parquet"


def test_matrix_roundtrip():
    x = np.random.default_rng(0).integers(0, 2, (5, 7)).astype(np.uint8)
    assert (column_matrix(matrix_column(x)) == x).all()


@pytest.mark.skipif(not TRAIN.exists(), reason="needs data/train.parquet")
def test_bin_spectra_on_real_data():
    table = pq.ParquetFile(TRAIN).read_row_group(0).slice(0, 500)
    table = table.filter(pc.is_in(table["adduct"], pa.array(["[M+H]+", "[M-H]-"])))
    x = bin_spectra(table)
    assert x.shape == (table.num_rows, MZ_BINS + LOSS_BINS + 10 + 3)
    assert np.isfinite(x).all()
    assert np.allclose(np.linalg.norm(x[:, :MZ_BINS], axis=1), 1, atol=1e-3)
