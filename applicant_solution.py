import json

import gdown
import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve, firwin

from task_and_baseline import baseline, build_task_helpers

url = "https://drive.google.com/file/d/1BBHVSI4KB-B8OX46eN1Nm4ARCeq6Rui4/view?usp=sharing"
downloaded_file = "challenge.mat"
gdown.download(url, downloaded_file, quiet=False, fuzzy=True)

data = loadmat("challenge.mat", simplify_cells=True)
tx   = data["tx"].astype(np.complex128)
rx   = data["rx"].astype(np.complex128)
Fs   = float(data["Fs"])

N, _ = tx.shape

tx_n = tx / (np.sqrt(np.mean(np.abs(tx) ** 2, axis=0, keepdims=True)) + 1e-30)

helpers = build_task_helpers(tx_n, Fs, N)



def your_canceller(tx_n, rx, helpers, n_iter=3):
    del tx_n

    score_filter = helpers["score_filter"]
    fit_tx_prediction = helpers["fit_tx_prediction"]

    def rank1_from_band_matrix(band_matrix, _eigh=np.linalg.eigh):
        cov = band_matrix.conj().T @ band_matrix / band_matrix.shape[0]
        _, vecs = _eigh(cov)
        shared = band_matrix @ vecs[:, -1]
        denom = np.vdot(shared, shared) + 1e-30
        return np.column_stack(
            [
                (np.vdot(shared, band_matrix[:, ch]) / denom) * shared
                for ch in range(band_matrix.shape[1])
            ]
        )

    tx_total = fit_tx_prediction(rx)
    r1_total = np.zeros_like(rx)

    def band_matrix(x):
        return np.column_stack(
            [
                score_filter(x[:, ch])
                for ch in range(x.shape[1])
            ]
        )

    def band_power(x):
        b = band_matrix(x)
        return np.mean(np.abs(b) ** 2)

    for it in range(n_iter):
        rx_for_r1 = rx - tx_total
        r1_total = rank1_from_band_matrix(band_matrix(rx_for_r1))

        rx_for_tx = rx - r1_total
        tx_total = fit_tx_prediction(rx_for_tx)

        rx_hat = rx - tx_total - r1_total
        db = 10 * np.log10(band_power(rx) / (band_power(rx_hat) + 1e-30))
        print(f"  [iter {it + 1}/{n_iter}]  total band removed: {db:.2f} dB")

    return rx - tx_total - r1_total

print("\n=== Baseline ===")
baseline_reds, baseline_avg = helpers["score"](
    rx, baseline(tx_n, rx, helpers["fit_tx_prediction"]), label="baseline"
)

print("=== Your Solution (iterative TX + rank-1, valid decomposition) ===")
yours_reds, yours_avg = helpers["score"](rx, your_canceller(tx_n, rx, helpers), label="yours")

results = {
    "baseline": {
        "per_channel_db": baseline_reds,
        "average_db":     baseline_avg,
    },
    "yours": {
        "per_channel_db": yours_reds,
        "average_db":     yours_avg,
    },
}

with open("results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved results.json")
