# 01 — Setup and data

## Requirements
- Windows 10/11 x64, ≥ 8 GB RAM (one unit uses < 3 GB), 3 GB free disk (venv 0.6 GB, data 0.5 GB, results < 1 GB).
- Internet on the first run (uv, Python packages, data, embedding models). Nothing is installed system-wide.
- No API key or token. (The OpenAI/PaLM embeddings of Beauty are the files the original authors published.)

## Installation
`run.bat` does everything: finds `uv` (or downloads it into `.tools\`), sets every cache (`UV_CACHE_DIR`, `HF_HOME`,
`FASTEMBED_CACHE_PATH`, `MPLCONFIGDIR`) inside the project, and runs `uv run --frozen` so the environment is exactly
`uv.lock` (Python 3.11.x, tensorflow 2.13.0 + tensorflow-intel, pandas 1.5.3, numpy 1.24.3, scikit-learn 1.3.2,
fastembed + onnxruntime 1.19.2).

Manual alternative: `uv sync --frozen` then `uv run --frozen python 99_master_run.py --smoke`.

## Data (`data_manifest.json`, downloaded by `00_kiem_tra_moi_truong.py` into `DataOG/`, read-only)
| Dataset | Source | Licence / terms | Files (size, MD5 checked on every launch) |
|---|---|---|---|
| Amazon Beauty 2014 (authors' pre-processed version) | github.com/dh-r/LLM-Sequential-Recommendation @ 6464b39 (Git LFS media URLs) | repository BSD-3-Clause; data: cite McAuley et al. | `dataset.pickle`, `products_lookup.csv`, `product_embeddings_openai.csv.gz`, `product_embeddings_palm.csv.gz` |
| MovieLens-1M | files.grouplens.org/datasets/movielens/ml-1m.zip | research use only, no redistribution, no commercial use | `ratings.dat`, `movies.dat`, `users.dat`, `README` |

Splits (`01`): Beauty keeps the authors' `TemporalSplit(test_frac=0.2, num_folds=3)` exactly (pickle unchanged).
MovieLens-1M: one user = one session, ordered by timestamp, the same `TemporalSplit` (the last 20 % of sessions by time are
test sessions; the last item of each test session is the target). This is **not** the leave-one-out protocol common in
other ML-1M papers, so absolute numbers are not comparable with them.

Item texts (`dataset/<ds>_items.csv`): Beauty = product name (as in the paper); ML-1M = `Title (Year). Genres: a, b`.

## Common errors
| Symptom | Cause / fix |
|---|---|
| `'run.bat' is not recognized` | the shell does not search the current folder (`NoDefaultCurrentDirectoryInExePath` is set): type `.\run.bat` or double-click it |
| `Bien UV_CACHE_DIR ... chua tro vao cache` | started without `run.bat`; use `run.bat` (it sets the cache variables) |
| `KHONG TAI DUOC <url>` | network; run again (download resumes per file); no synthetic data is ever used instead |
| `TU CHOI: RAM trong ... < 2 GB` | close other programs |
| access violation in onnxruntime | onnxruntime newer than 1.19.2 with numpy 1.24 — keep `uv.lock` (`--frozen`) |
| window red + beep | a unit failed: last `CRASH` entry in `Result\master_log.txt`; fix and run the same command again |
