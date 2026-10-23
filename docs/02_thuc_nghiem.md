# 02 — Experiments: what, why, where

All models come from the authors' code (`vendor/llmseqrec`) with the tuned configurations they hard-coded in
`run_experiments.ipynb`; `models.py` lists them. Protocol (unchanged from the original): next-item prediction for the
last item of each test session, ranking over the full catalogue, items of the prompt excluded, early stopping on an inner
10 % validation split (at most 500 sessions, patience 2, ≤ 100 epochs). Metrics: NDCG@10/20, HitRate@10/20, MRR@10/20, catalogue coverage@20.

| Unit (`03`, `--model`) | What it is | Why it is in the study |
|---|---|---|
| `random` | uniform random ranking | chance level (sanity check: Popular > Random) |
| `popular`, `vsknn` | most-popular; vector session-kNN | non-neural reference points |
| `gru4rec`, `sasrec`, `bert4rec` | ID-only sequential models (original tuned configs) | "compare with another model" |
| `sasrec_l2s` | SASRec with **LLM2SASRec's configuration**, random item init | separates the effect of the LLM initialisation from the effect of the hyper-parameters (H3) |
| `llm2sasrec`, `llm2bert4rec` | item table initialised from LLM embeddings (PCA; LDA for PaLM+BERT as in the original) | the method being replicated (H1) and re-sourced with open embeddings (H4, H5) |
| `llm2sasrec_shuf` | PaLM table with rows permuted (replicates Harte 2023 §4.2) | semantics vs distribution of the initial table |
| `fused` | frozen L2-normalised LLM table → linear adapter + zero-initialised ID embedding | revision of the model: fusion instead of initialisation (H6, ablation) |
| `llmseqsim` | session = harmonic-decay mean of item embeddings, cosine kNN | the authors' training-free LLM method; cheap test of embedding sources |

Embedding sources (`--emb`): `openai` (ada-002) and `palm` — the authors' published files (Beauty only); `bge-base`,
`bge-small`, `minilm` — open-source, computed locally by `02`.

Schedule (`99_master_run.py`, mirrors ho_so/09_exp_plan.md): tier A = the core comparisons (5 seeds), tier B = other embedding
sources, Fused, GRU4Rec (5 seeds), V-SKNN (deterministic, 1 run), tier C = BERT-family (3 seeds, expensive on CPU).

Statistics (`04`, `stats.py`): primary test = paired t-test over seeds (95 % t CI), Holm over the four pre-specified
comparisons C1–C4. Secondary (describes the trained models only): per-session Wilcoxon signed-rank and bootstrap 95 % CI.
Non-inferiority decision row: LLM2SASRec BGE-base vs PaLM with δ = 10 % of the PaLM mean (other embedder pairs descriptive); "best-of-k vs mean" table.

Where the results are:
- every unit: `Result/<run>/` — `config.json`, `log.csv`, `metrics.csv`, `figures/`, `data/` (`per_session.csv`, `history.csv`, `top20.csv.gz` for seed 42);
- the book of all runs: `Result/ThucNghiem_Log.csv` (experiment code → folder);
- analysis: the latest `Result/*_ANA-ALL-01_*/data/bang_4-0*.csv` and its `figures/`;
- web demo data: `../app_web/public/data/` (written by `05`).
