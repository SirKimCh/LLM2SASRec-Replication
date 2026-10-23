"""03 — one experimental unit = one process: train + predict + evaluate one (dataset, model, embedding, seed).
One process per unit is required: the original SASRecWithEmbeddings caches the embedding file in a CLASS variable.
Evaluation = the authors' code (main.eval) at top-20/top-10 on the TemporalSplit test sessions, full catalogue,
prompt items excluded. Per-session NDCG@20 is recomputed here for the paired tests and checked against the authors' value."""
import argparse, hashlib, json, time

import numpy as np
import pandas as pd

import lib

CODES = {"popular": 1, "vsknn": 2, "gru4rec": 3, "sasrec": 4, "sasrec_l2s": 5, "bert4rec": 6, "llmseqsim": 7, "llm2sasrec": 8,
         "llm2bert4rec": 9, "fused": 10, "random": 11, "llm2sasrec_shuf": 12}


def unit_key(ds, model, emb, seed):
    return f"rec:{ds}:{model}:{emb}:{seed}"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def per_session(preds, gts):
    """NDCG@20 / Hit@20 / rank per test session; one held-out item per session (n_withheld=1)."""
    rows = []
    for sid, gt in gts.items():
        rec = list(preds.get(sid, []))[:20]
        g = gt[0] if len(gt) else None
        r = rec.index(g) if g in rec else -1
        rows.append((sid, 1 / np.log2(r + 2) if r >= 0 else 0.0, int(r >= 0), r + 1 if r >= 0 else 0))
    return pd.DataFrame(rows, columns=["SessionId", "ndcg20", "hit20", "rank"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=lib.DATASETS)
    ap.add_argument("--model", required=True)
    ap.add_argument("--emb", default="-")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import tensorflow as tf
    lib.set_seed(a.seed)
    import models
    from keras.callbacks import Callback
    from main.data.session_dataset import SessionDataset
    from main.eval.evaluation import Evaluation, metrics
    from main.eval.metrics import NormalizedDiscountedCumulativeGain, HitRate, MeanReciprocalRank, CatalogCoverage

    name = models.NAMES[a.model]
    ma = f"REC-{a.ds.upper()}-{CODES[a.model]:02d}"
    run = lib.new_run(ma, name, f"{a.emb}-s{a.seed}", "smoke" if a.smoke else "full", smoke=a.smoke, unit=unit_key(a.ds, a.model, a.emb, a.seed))
    log = lib.RunLog(run)
    pk = lib.ds_dir(a.smoke) / f"{a.ds}.pickle"
    ds = SessionDataset.from_pickle(str(pk))
    train, prompts, gts = ds.get_train_data(), ds.get_test_prompts(), ds.get_test_ground_truths()
    emb_path, dim, item_data = None, None, None
    if a.model in models.NEEDS_EMB:
        emb_path = lib.ds_dir(a.smoke) / f"{a.ds}_emb_{a.emb}.csv.gz"
        if not emb_path.exists():
            raise SystemExit(f"thieu {emb_path} (chay 01/02 truoc)")
        if a.model == "llmseqsim":
            item_data = pd.read_csv(emb_path)
            item_data["embedding"] = [np.array(json.loads(e), dtype=np.float32) for e in item_data["embedding"]]
            dim = len(item_data["embedding"].iloc[0])
    epochs = 1 if a.smoke else 100
    model, cfg = models.build(a.model, str(emb_path) if emb_path else None, a.emb, lib.plan()["cores"], epochs, dim, a.seed)

    hist = []

    class Hist(Callback):
        def on_epoch_end(self, epoch, logs=None):
            hist.append({"epoch": epoch + 1, **{k: float(v) for k, v in (logs or {}).items()}})
            log.event("epoch", json.dumps(hist[-1]))

    if hasattr(model, "active_callbacks"):
        model.active_callbacks.append(Hist())
    log.event("start", f"{name} {a.ds} emb={a.emb} seed={a.seed} epochs<={epochs}")
    t = time.time()
    model.train(train, item_data) if a.model == "llmseqsim" else model.train(train)
    train_s = time.time() - t
    log.event("train", f"{len(hist)} epoch", sec=train_s)
    t = time.time()
    preds = model.predict(prompts, top_k=20)
    pred_s = time.time() - t
    log.event("predict", f"{len(preds)} phien", sec=pred_s)

    deps = {metrics.MetricDependency.NUM_ITEMS: ds.get_unique_item_count()}
    res = {}
    for k in (10, 20):
        rep = Evaluation.eval(predictions=preds, ground_truths=gts, model_name=name, top_k=k, dependencies=deps, metrics_per_sample=False,
                              metrics=[NormalizedDiscountedCumulativeGain(), HitRate(), MeanReciprocalRank(), CatalogCoverage()])
        res.update(rep.to_df().to_dict("records")[0])
    ps = per_session(preds, gts)
    if abs(ps.ndcg20.mean() - res["NDCG@20"]) > 1e-6:          # self-check: our per-session values reproduce the authors' metric
        raise SystemExit(f"per-session NDCG@20 {ps.ndcg20.mean():.6f} != NDCG@20 cua code goc {res['NDCG@20']:.6f}")
    ps.to_csv(run / "data" / "per_session.csv", index=False)
    if a.seed == 42:                                           # top-20 lists for the web demo (one seed is enough)
        pd.DataFrame([(s, i + 1, int(it)) for s, rec in preds.items() for i, it in enumerate(list(rec)[:20])],
                     columns=["SessionId", "rank", "ItemId"]).to_csv(run / "data" / "top20.csv.gz", index=False)
    with open(run / "data" / "data_manifest.csv", "w", encoding="utf-8") as f:
        f.write("file,md5\n" + f"dataset/{pk.name},{md5(pk)}\n" + (f"dataset/{emb_path.name},{md5(emb_path)}\n" if emb_path else ""))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if hist:
        h = pd.DataFrame(hist)
        h.to_csv(run / "data" / "history.csv", index=False)
        fig, ax1 = plt.subplots(figsize=(5, 3.2))
        ax1.plot(h.epoch, h["loss"], "o-", color="#4C72B0", label="loss"); ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
        if "inner_val_NDCG@20" in h:
            ax2 = ax1.twinx(); ax2.plot(h.epoch, h["inner_val_NDCG@20"], "s--", color="#C44E52"); ax2.set_ylabel("inner-val NDCG@20")
        ax1.set_title(f"{name} / {a.ds} / {a.emb} / seed {a.seed}")
        fig.tight_layout(); fig.savefig(run / "figures" / f"fig_4-01_{a.ds}_{a.model}_hoc.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(ps.loc[ps["rank"] > 0, "rank"], bins=np.arange(1, 22) - 0.5, color="#55A868")
    ax.set_xlabel("vi tri cua item dung trong top-20"); ax.set_ylabel("so phien"); ax.set_title(f"{name}: HR@20 = {res['HitRate@20']:.4f}")
    fig.tight_layout(); fig.savefig(run / "figures" / f"fig_4-02_{a.ds}_{a.model}_vi-tri-trung.png", dpi=150); plt.close(fig)

    import sklearn, scipy
    lib.write_config(run, {"dataset": a.ds, "model": name, "model_key": a.model, "embedding": a.emb, "seed": a.seed, "smoke": a.smoke,
                           "config": cfg, "epochs_run": len(hist), "vendor_commit": lib.VENDOR_COMMIT,
                           "versions": {"tensorflow": tf.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                                        "sklearn": sklearn.__version__, "scipy": scipy.__version__},
                           "protocol": "TemporalSplit test, last item, full catalogue, pred_seen False, early stopping (patience 2) on the original inner validation split: 10 % of train sessions capped at 500"})
    rows = [(k, v, "test", "next_item", name) for k, v in res.items()]
    rows += [("train_sec", train_s, "train", "time", name), ("predict_sec", pred_s, "test", "time", name), ("epochs_run", len(hist), "train", "time", name),
             ("test_sessions", len(ps), "test", "count", name)]
    lib.write_metrics(run, rows)
    lib.log_book(a.smoke, ma, a.ds, name, f"emb={a.emb};seed={a.seed}", run, "huan luyen + danh gia (code goc)", True)
    print(f"[03] {a.ds} {name} {a.emb} s{a.seed}: NDCG@20 {res['NDCG@20']:.4f} HR@20 {res['HitRate@20']:.4f} ({train_s:.0f}s train, {len(hist)} ep)", flush=True)


if __name__ == "__main__":
    main()
