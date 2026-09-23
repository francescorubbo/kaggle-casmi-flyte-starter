"""A CPU retrieval baseline that turns each molecular representation into a local MRR@25.

    spectrum --(bin)--> x --(MLP)--> predicted representation --(similarity)--> ranked candidates

Candidates are every library molecule whose monoisotopic mass matches the precursor-derived
neutral mass (within `ppm`). The hold-out molecules are in the library but none of their spectra
were used for training: the "class 2" setting (known structure, no public spectra).
"""

import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from flyte.io import File

from casmi_flyte.config import TEST_ADDUCTS, neutral_mass
from casmi_flyte.metric import K, mrr_at_k
from casmi_flyte.tables import column_matrix, read_table

MZ_BINS = 1200  # 1 Da fragment bins, test precursors go up to ~1160
LOSS_BINS = 300  # neutral losses (precursor - fragment) up to 300 Da
ADDUCTS = list(TEST_ADDUCTS)


def bin_spectra(table: pa.Table, min_rel_intensity: float = 0.01) -> np.ndarray:
    """Vectorized: fragment + neutral-loss histograms (sqrt intensity), adduct one-hot, energy, mass."""
    n = table.num_rows
    mzs = table["ms2_mzs"].combine_chunks()
    row = pc.list_parent_indices(mzs).to_numpy()
    mz = pc.list_flatten(mzs).to_numpy()
    inten = pc.list_flatten(table["ms2_normalized_intensities"].combine_chunks()).to_numpy()
    prec = table["precursor_mz"].to_numpy()

    keep = (inten >= min_rel_intensity) & (mz <= prec[row] + 2)
    row, mz, inten = row[keep], mz[keep], np.sqrt(inten[keep])

    frag = np.zeros((n, MZ_BINS), dtype=np.float32)
    np.add.at(frag, (row, np.clip(np.rint(mz).astype(int), 0, MZ_BINS - 1)), inten)
    loss_mz = prec[row] - mz
    ok = (loss_mz > 0.5) & (loss_mz < LOSS_BINS - 0.5)
    loss = np.zeros((n, LOSS_BINS), dtype=np.float32)
    np.add.at(loss, (row[ok], np.rint(loss_mz[ok]).astype(int)), inten[ok])
    for m in (frag, loss):  # per-spectrum L2 normalisation
        m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-8

    adduct = np.zeros((n, len(ADDUCTS)), dtype=np.float32)
    idx = pc.index_in(table["adduct"], pa.array(ADDUCTS)).to_numpy(zero_copy_only=False)
    adduct[np.arange(n), idx] = 1
    ce = np.array([np.mean(v) if v else np.nan for v in table["collision_energy_ev"].to_pylist()], dtype=np.float32)
    extra = np.stack([np.nan_to_num(np.log1p(ce) / 5), np.isnan(ce), prec / 1000], axis=1).astype(np.float32)
    return np.hstack([frag, loss, adduct, extra])


def _neutral_masses(table: pa.Table) -> np.ndarray:
    return np.array(
        [neutral_mass(p, a) for p, a in zip(table["precursor_mz"].to_pylist(), table["adduct"].to_pylist())]
    )


def _similarity(pred: np.ndarray, cands: np.ndarray, binary: bool) -> np.ndarray:
    if binary:  # soft Tanimoto between predicted bit probabilities and candidate bits
        cands = cands.astype(np.float32)
        inter = cands @ pred
        return inter / (pred.sum() + cands.sum(axis=1) - inter + 1e-8)
    return (cands @ pred) / (np.linalg.norm(cands, axis=1) * np.linalg.norm(pred) + 1e-8)


# Not a task yet: decide which environment (image, resources) this should run in and decorate it.
# It needs numpy, pyarrow and scikit-learn; `masses` must provide `inchikey14`, `exact_mass`, `valid`.
async def evaluate_representation(
    representation: str,
    features: File,
    masses: File,
    train_spectra: File,
    holdout_spectra: File,
    max_train_spectra: int = 100_000,
    epochs: int = 8,
    ppm: float = 10.0,
) -> dict[str, float]:
    """Train spectrum -> `representation` and score retrieval on the hold-out. Returns metrics."""
    from sklearn.neural_network import MLPClassifier, MLPRegressor

    t0 = time.time()
    lib = await read_table(features, columns=["inchikey14", "valid", representation])
    lib = lib.filter(lib["valid"])
    mass_table = await read_table(masses, columns=["inchikey14", "exact_mass", "valid"])
    mass_table = mass_table.filter(mass_table["valid"])
    # (pyarrow joins do not support fixed-size-list payload columns: look the masses up instead)
    mass_of = dict(zip(mass_table["inchikey14"].to_pylist(), mass_table["exact_mass"].to_pylist()))
    lib_mass_raw = np.array([mass_of.get(k, np.nan) for k in lib["inchikey14"].to_pylist()])
    lib = lib.filter(pa.array(~np.isnan(lib_mass_raw))).append_column(
        "exact_mass", pa.array(lib_mass_raw[~np.isnan(lib_mass_raw)])
    )
    Y_lib = column_matrix(lib[representation])
    binary = Y_lib.dtype == np.uint8  # fingerprints are stored as uint8 bits, descriptors as float32
    if not binary:  # standardise continuous features so that no descriptor dominates the cosine
        Y_lib = np.nan_to_num(Y_lib.astype(np.float32))
        mu, sd = Y_lib.mean(0), Y_lib.std(0) + 1e-6
        Y_lib = (Y_lib - mu) / sd
    key_to_row = {k: i for i, k in enumerate(lib["inchikey14"].to_pylist())}
    print(f"{representation}: library of {len(key_to_row):,} molecules x {Y_lib.shape[1]} ({'bits' if binary else 'floats'})")

    # --- training --------------------------------------------------------------------------
    train = await read_table(train_spectra)
    rows = np.array([key_to_row.get(k, -1) for k in train["inchikey14"].to_pylist()])
    train = train.filter(pa.array(rows >= 0))
    rows = rows[rows >= 0]
    if train.num_rows > max_train_spectra:
        pick = np.sort(np.random.default_rng(0).choice(train.num_rows, max_train_spectra, replace=False))
        train, rows = train.take(pa.array(pick)), rows[pick]
    X = bin_spectra(train)
    Y = Y_lib[rows]
    print(f"training on {X.shape[0]:,} spectra, {X.shape[1]} features -> {Y.shape[1]} targets")
    Model = MLPClassifier if binary else MLPRegressor
    model = Model(hidden_layer_sizes=(512,), batch_size=256, max_iter=epochs, learning_rate_init=1e-3, random_state=0)
    model.fit(X, Y.astype(np.int8) if binary else Y)
    t_train = time.time() - t0

    # --- retrieval on the hold-out -----------------------------------------------------------
    hold = await read_table(holdout_spectra)
    Xh = bin_spectra(hold)
    P = model.predict_proba(Xh) if binary else model.predict(Xh)
    M = _neutral_masses(hold)
    keys = np.array(hold["inchikey14"].to_pylist())

    order = np.argsort(lib["exact_mass"].to_numpy())
    lib_mass = lib["exact_mass"].to_numpy()[order]
    lib_keys = np.array(lib["inchikey14"].to_pylist())[order]
    Y_sorted = Y_lib[order]

    rng = np.random.default_rng(0)
    predictions, random_predictions, n_cands, answers, in_window = {}, {}, [], {}, []
    for mol in np.unique(keys):
        sel = keys == mol
        m = float(np.median(M[sel]))
        lo, hi = np.searchsorted(lib_mass, [m * (1 - ppm * 1e-6), m * (1 + ppm * 1e-6)])
        n_cands.append(hi - lo)
        in_window.append(mol in set(lib_keys[lo:hi]))
        answers[mol] = mol  # pseudo molecule_id = its InChIKey14
        if hi == lo:
            continue
        sim = _similarity(P[sel].mean(axis=0), Y_sorted[lo:hi], binary)
        predictions[mol] = list(lib_keys[lo:hi][np.argsort(-sim)[:K]])
        random_predictions[mol] = list(rng.permutation(lib_keys[lo:hi])[:K])

    top1 = np.mean([predictions.get(m, [None])[0] == m for m in answers])
    metrics = {
        "mrr@25": mrr_at_k(predictions, answers),
        "top1": float(top1),
        "random_mrr@25": mrr_at_k(random_predictions, answers),
        "n_holdout_molecules": float(len(answers)),
        "median_candidates": float(np.median(n_cands)),
        "answer_in_top25": float(np.mean([m in predictions.get(m, []) for m in answers])),
        "answer_in_mass_window": float(np.mean(in_window)),
        "train_spectra": float(X.shape[0]),
        "train_seconds": t_train,
    }
    print(metrics)
    return metrics
