"""02 — open-source text embeddings of the item texts with fastembed (ONNX, CPU, no API key).
Output dataset/<ds>_emb_<emb>.csv.gz in the authors' format (ItemId, embedding as a JSON list), one row per item.
Run EMB-<DS>-<emb>: metrics (items, dim, seconds) + figures (cosine of random pairs, 2-D PCA coloured by popularity)."""
import argparse, json, os, time

import numpy as np
import pandas as pd

import lib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=lib.DATASETS)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    man = json.loads((lib.ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    model = man["embedders"][a.emb]
    items = pd.read_csv(lib.ds_dir(a.smoke) / f"{a.ds}_items.csv").fillna({"text": ""})
    out = lib.ds_dir(a.smoke) / f"{a.ds}_emb_{a.emb}.csv.gz"
    run = lib.new_run(f"EMB-{a.ds.upper()}-01", a.emb, "fastembed-onnx", "item-text", smoke=a.smoke, unit=f"emb:{a.ds}:{a.emb}")
    log = lib.RunLog(run)
    from fastembed import TextEmbedding
    t = time.time()
    enc = TextEmbedding(model_name=model, threads=lib.plan()["cores"], cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
    log.event("load", model, sec=time.time() - t)
    t = time.time()
    vecs = np.array(list(enc.embed(items.text.tolist(), batch_size=64)), dtype=np.float32)
    sec = time.time() - t
    if len(vecs) != len(items) or not np.isfinite(vecs).all():
        raise SystemExit(f"02: embedding thieu hoac NaN ({len(vecs)}/{len(items)})")
    pd.DataFrame({"ItemId": items.ItemId, "embedding": [json.dumps(v.round(6).tolist()) for v in vecs]}).to_csv(out, index=False)
    log.event("embed", f"{len(vecs)} x {vecs.shape[1]}", sec=sec)
    # figures
    rng = np.random.default_rng(42)
    u = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    i, j = rng.integers(0, len(u), 5000), rng.integers(0, len(u), 5000)
    cos = (u[i] * u[j]).sum(1)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(cos, bins=60, color="#4C72B0"); ax.set_xlabel("cosine giua 2 item ngau nhien"); ax.set_title(f"{a.ds} / {a.emb}")
    fig.tight_layout(); fig.savefig(run / "figures" / f"fig_3-06_{a.ds}_{a.emb}_cosine.png", dpi=150); plt.close(fig)
    from main.data.session_dataset import SessionDataset
    pop = SessionDataset.from_pickle(str(lib.ds_dir(a.smoke) / f"{a.ds}.pickle")).get_train_data().ItemId.value_counts()
    c = u - u.mean(0)
    xy = c @ np.linalg.svd(c, full_matrices=False)[2][:2].T
    col = np.log1p(items.ItemId.map(pop).fillna(0).values)
    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=col, s=2, cmap="viridis"); fig.colorbar(sc, label="log(1+so tuong tac)")
    ax.set_title(f"{a.ds} / {a.emb}: PCA 2 chieu"); fig.tight_layout()
    fig.savefig(run / "figures" / f"fig_3-07_{a.ds}_{a.emb}_pca2d.png", dpi=150); plt.close(fig)
    with open(run / "data" / "data_manifest.csv", "w", encoding="utf-8") as f:
        f.write(f"file,rows,dim\n{out.name},{len(vecs)},{vecs.shape[1]}\n")
    lib.write_config(run, {"dataset": a.ds, "embedder": model, "library": "fastembed", "output": f"dataset/{out.name}", "text_field": "items.csv:text"})
    lib.write_metrics(run, [("items", len(vecs), "all", "embedding", a.emb), ("dim", vecs.shape[1], "all", "embedding", a.emb),
                            ("embed_sec", sec, "all", "embedding", a.emb), ("cos_random_mean", float(cos.mean()), "all", "embedding", a.emb)])
    lib.log_book(a.smoke, f"EMB-{a.ds.upper()}-01", a.ds, a.emb, "fastembed-onnx", run, f"embedding van ban {model}", True)
    print(f"[02] {a.ds}/{a.emb}: {vecs.shape} trong {sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
