# LLM-enhanced sequential recommendation — replication & extension on the original code

Course project (Intelligent Information Systems, Can Tho University — group 9: Nguyễn Thế Trung, Trần Vạn Thành;
supervisor Assoc. Prof. Nguyễn Thái Nghe). Topic 1 of the course: *select a method from
[dh-r/LLM-Sequential-Recommendation](https://github.com/dh-r/LLM-Sequential-Recommendation), re-run the code, revise the
model, replace the dataset, compare with another model.*

The experiments answer, **on the authors' own code** (Harte et al., RecSys'23; Boz et al., ACM TORS 2025):

1. Does LLM2SASRec (SASRec whose item-embedding table is initialised from LLM text embeddings reduced by PCA) still beat
   SASRec when results are reported as mean ± std over 5 seeds instead of the best of 5 runs selected on the test set?
2. Which part of the gain comes from the embedding initialisation and which part from the hyper-parameters
   (SASRec trained with LLM2SASRec's configuration but random initialisation)?
3. Can open-source local text embeddings (BGE, MiniLM) replace the commercial OpenAI ada-002 / PaLM embeddings?
4. Does the effect transfer to a new dataset, MovieLens-1M?
5. Revision of the model: keep the LLM embedding **frozen** behind a learned linear adapter plus a trainable ID embedding
   (Fused-SASRec, an ablation in the spirit of LLM-ESR) instead of initialisation only.

Baselines from the same code base: Popular, V-SKNN, GRU4Rec, SASRec, BERT4Rec, LLM2BERT4Rec, LLMSeqSim.

## Quickstart (Windows)

```bat
run.bat --smoke      :: about 10 min end-to-end self-test (1 seed, 1 epoch) into Result_smoke\ (deleted afterwards; --keep to keep it)
run.bat --estimate   :: time estimate for the full schedule on this machine
run.bat --tier A     :: core comparisons only (about 2.1 h on a laptop CPU, estimate of `--estimate`)
run.bat              :: everything (tiers A, B, C; 116 units, estimate 9.2 h, measured 8.6 h on an i5-12450HX CPU). Resumes after any interruption: just run it again.
clean.bat            :: after copying Result\ away: removes .venv, data and caches, keeps code + Result\
```

`run.bat` finds or downloads `uv`, keeps all caches inside the project, builds `.venv` exactly from `uv.lock`
(Python 3.11, TensorFlow 2.13 CPU — the version the original code needs), then `99_master_run.py`
downloads the data (MD5-checked) and runs the schedule. No API key is needed.

Hardware: any x86-64 Windows machine with ≥ 8 GB RAM and 3 GB free disk. TensorFlow 2.13 on Windows runs on the CPU;
a GPU is not used.

## Layout

```
src/
├── run.bat · clean.bat           one command to run / to clean the machine
├── 00_kiem_tra_moi_truong.py     environment + data check, downloads DataOG/ (data_manifest.json, MD5)
├── 01_chuan_bi_du_lieu.py        DataOG/ -> dataset/ (Beauty pickle of the authors; ML-1M with the same TemporalSplit) + EDA
├── 02_tao_embedding.py           open-source text embeddings (fastembed, ONNX, CPU)
├── 03_chay_mo_hinh.py            one unit = one (dataset, model, embedding, seed): train, predict, evaluate
├── 04_phan_tich.py               tables, paired t over seeds (primary, Holm over C1–C4); per-session Wilcoxon + bootstrap (secondary), figures
├── 05_xuat_du_lieu_web.py        exports the results for the Next.js demo app (../app_web)
├── 99_master_run.py              schedule, resume, --smoke, --estimate, --budget-hours, progress line
├── lib.py · models.py · stats.py helpers, model registry (original configurations), statistics
├── vendor/llmseqrec/             the authors' code (main/*.py, commit 6464b39, BSD-3), unchanged
├── tests/                        pytest: resume, run folders, tier order, clean.bat, statistics
├── docs/                         setup and experiment documentation
└── Result/                       every result: run folders, ThucNghiem_Log.csv, run_state.json, master_log.txt
```

| Document | Content |
|---|---|
| [docs/01_setup.md](docs/01_setup.md) | installation, data sources and licences, common errors |
| [docs/02_thuc_nghiem.md](docs/02_thuc_nghiem.md) | what each experiment does and why, where its results are |

Data licences: Amazon Beauty as shared by the original authors (cite McAuley et al.); MovieLens-1M
(GroupLens — research use only, **no redistribution**: the data are downloaded by `00`, never shipped with this code).
