"""05 — export the real results for the Next.js app (data contract in ../app_web/README.md):
manifest.json, results.json, <ds>/items.json, <ds>/emb_<id>.bin (Float32, PCA to 192 dims then L2-normalised, so the app stays
small), <ds>/demo_sessions.json (200 test sessions with the seed-42 top-10 of several models).
Writes into ../app_web/public/data when that folder exists (or --out), else into the run's data/web/. Smoke never touches the app."""
import argparse, gzip, json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import lib
from models import NAMES

GROUP = {"random": "Non-neural", "llm2sasrec_shuf": "LLM-enhanced", "popular": "Non-neural", "vsknn": "Non-neural", "gru4rec": "ID-only", "sasrec": "ID-only", "sasrec_l2s": "ID-only", "bert4rec": "ID-only",
         "llm2sasrec": "LLM-enhanced", "llm2bert4rec": "LLM-enhanced", "llmseqsim": "LLM-enhanced", "fused": "Ours"}
DS_NAME = {"beauty": "Amazon Beauty", "ml1m": "MovieLens-1M"}
EMB_LABEL = {"bge-base": "BGE-base-en-v1.5 (mã nguồn mở)", "openai": "OpenAI text-embedding-ada-002"}
DEMO_MODELS = [("sasrec", "-"), ("llm2sasrec", "openai"), ("llm2sasrec", "bge-base"), ("fused", "bge-base"), ("llmseqsim", "bge-base")]
DIM = 192


def reduce(E):
    E = E / np.linalg.norm(E, axis=1, keepdims=True).clip(min=1e-12)
    C = E - E.mean(0)
    Z = C @ np.linalg.svd(C, full_matrices=False)[2][:DIM].T
    return (Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-12)).astype("<f4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = lib.new_run("WEB-ALL-01", "xuat-web", f"pca{DIM}", "app-data", smoke=a.smoke, unit="web")
    log = lib.RunLog(run)
    app = lib.ROOT.parent / "app_web" / "public" / "data"
    out = Path(a.out) if a.out else (app if app.exists() and not a.smoke else run / "data" / "web")
    out.mkdir(parents=True, exist_ok=True)
    from main.data.session_dataset import SessionDataset
    runs = lib.runs(a.smoke)
    manifest = {"mock": False, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "datasets": []}
    sizes = []
    for ds in lib.DATASETS:
        items = pd.read_csv(lib.ds_dir(a.smoke) / f"{ds}_items.csv").fillna({"text": ""})
        titles = items.text.str.split(". Genres:").str[0] if ds == "ml1m" else items.text
        (out / ds).mkdir(exist_ok=True)
        (out / ds / "items.json").write_text(json.dumps([{"id": int(i), "title": t} for i, t in zip(items.ItemId, titles)], ensure_ascii=False), encoding="utf-8")
        embs = []
        for e in ("bge-base", "openai"):
            p = lib.ds_dir(a.smoke) / f"{ds}_emb_{e}.csv.gz"
            if not p.exists():
                continue
            df = pd.read_csv(p).set_index("ItemId").reindex(items.ItemId)
            if df.embedding.isna().any():
                raise SystemExit(f"05: {p.name} thieu embedding cho {int(df.embedding.isna().sum())} item")
            E = np.vstack([np.asarray(json.loads(x), dtype=np.float32) for x in df.embedding])
            (out / ds / f"emb_{e}.bin").write_bytes(reduce(E).tobytes())
            embs.append({"id": e, "label": EMB_LABEL[e], "dim": DIM, "file": f"{ds}/emb_{e}.bin"})
        d = SessionDataset.from_pickle(str(lib.ds_dir(a.smoke) / f"{ds}.pickle"))
        tr, prompts, gts = d.get_train_data(), d.get_test_prompts(), d.get_test_ground_truths()
        manifest["datasets"].append({"id": ds, "name": DS_NAME[ds], "n_items": len(items), "n_sessions": int(tr.SessionId.nunique() + len(prompts)),
                                     "n_interactions": int(len(tr) + sum(len(v) + 1 for v in prompts.values())), "embeddings": embs})
        recs = {}
        for dirp, cfg in runs:
            key = (cfg.get("model_key"), cfg.get("embedding"))
            if cfg.get("dataset") == ds and cfg.get("seed") == 42 and key in DEMO_MODELS and (dirp / "data" / "top20.csv.gz").exists():
                t = pd.read_csv(dirp / "data" / "top20.csv.gz")
                recs[NAMES[key[0]] + ("" if key[1] == "-" else f" ({key[1]})")] = t[t["rank"] <= 10].groupby("SessionId")["ItemId"].apply(list).to_dict()
        rng = np.random.default_rng(42)
        sids = list(prompts)
        hit = [s for s in sids if any(gts[s][0] in r.get(s, []) for r in recs.values())]
        pick = list(rng.choice(hit, min(100, len(hit)), replace=False)) if hit else []
        rest = [s for s in sids if s not in set(pick)]
        pick += list(rng.choice(rest, min(200 - len(pick), len(rest)), replace=False))
        demo = [{"sid": int(s), "history": [int(x) for x in prompts[s]][-30:], "truth": int(gts[s][0]),
                 "recs": {m: [int(x) for x in r.get(s, [])] for m, r in recs.items()}} for s in pick]
        (out / ds / "demo_sessions.json").write_text(json.dumps(demo), encoding="utf-8")
        log.event("web", f"{ds}: {len(items)} item, {len(embs)} embedding, {len(demo)} phien demo, {len(recs)} mo hinh")
    anas = sorted((d for d, c in runs if c.get("unit") == "ana"), key=lambda d: (d / "metrics.csv").stat().st_mtime)
    res = []
    if anas:
        sm = pd.read_csv(anas[-1] / "data" / "bang_4-01_tong_hop.csv")
        for r in sm.to_dict("records"):                    # itertuples would rename columns containing '@'
            res.append({"dataset": r["dataset"], "model": NAMES[r["model"]], "group": GROUP[r["model"]], "embedding": None if r["embedding"] == "-" else r["embedding"],
                        "metrics": {k: {"mean": float(r[f"{k}_mean"]), "std": float(np.nan_to_num(r[f"{k}_std"]))}
                                    for k in ("NDCG@10", "NDCG@20", "HitRate@10", "HitRate@20", "MRR@20")}, "n_seeds": int(r["n_seeds"])})
    (out / "results.json").write_text(json.dumps(res, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    sizes = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    lib.write_config(run, {"out": str(out.relative_to(lib.ROOT.parent)) if lib.ROOT.parent in out.parents else str(out), "pca_dim": DIM,
                           "demo_models": [list(m) for m in DEMO_MODELS], "source_ana": anas[-1].name if anas else None})
    with open(run / "data" / "data_manifest.csv", "w", encoding="utf-8") as f:
        f.write("file,bytes\n" + "".join(f"{p.relative_to(out)},{p.stat().st_size}\n" for p in sorted(out.rglob("*")) if p.is_file()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = sorted((p for p in out.rglob("*") if p.is_file()), key=lambda p: p.stat().st_size)
    fig, ax = plt.subplots(figsize=(5, 0.25 * len(files) + 1))
    ax.barh([str(p.relative_to(out)) for p in files], [p.stat().st_size / 1e6 for p in files], color="#2a78d6")
    ax.set_xlabel("MB"); ax.set_title("Du lieu xuat cho ung dung web"); fig.tight_layout()
    fig.savefig(run / "figures" / "fig_5-01_web_dung-luong.png", dpi=150); plt.close(fig)
    lib.write_metrics(run, [("bytes_total", sizes, "web", "app_data", "export"), ("results_rows", len(res), "web", "app_data", "export")])
    lib.log_book(a.smoke, "WEB-ALL-01", "beauty+ml1m", "xuat-web", f"pca{DIM}", run, "du lieu cho app Next.js", True)
    print(f"[05] {sizes / 1e6:.1f} MB -> {out}", flush=True)


if __name__ == "__main__":
    main()
