"""01 — the only step that reads DataOG/. Writes dataset/:
  beauty.pickle (the authors' SessionDataset, unchanged), beauty_items.csv (ItemId,text = product name, as in the paper),
  beauty_emb_openai.csv.gz / beauty_emb_palm.csv.gz (the authors' embedding files, unchanged),
  ml1m.pickle (one user = one session, TemporalSplit(test_frac=0.2, num_folds=3) exactly as the authors split Beauty),
  ml1m_items.csv (ItemId,text = "Title (Year). Genres: a, b").
Plus runs PRE-<DS>-01 (data description, before/after tables) and EDA-<DS>-01 (figures)."""
import argparse, os, shutil, time

import numpy as np
import pandas as pd

import lib
from main.data.session_dataset import SessionDataset, TemporalSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def outputs(smoke=False):
    d = lib.ds_dir(smoke)
    return [d / f for f in ("beauty.pickle", "beauty_items.csv", "beauty_emb_openai.csv.gz", "beauty_emb_palm.csv.gz", "ml1m.pickle", "ml1m_items.csv")]


def prep_beauty(out):
    src = lib.dataog() / "beauty"
    link_or_copy(src / "dataset.pickle", out / "beauty.pickle")
    for e in ("openai", "palm"):
        link_or_copy(src / f"product_embeddings_{e}.csv.gz", out / f"beauty_emb_{e}.csv.gz")
    p = pd.read_csv(src / "products_lookup.csv", usecols=["global_product_id", "name"])
    p.rename(columns={"global_product_id": "ItemId", "name": "text"}).fillna({"text": ""}).to_csv(out / "beauty_items.csv", index=False)


def prep_ml1m(out):
    src = lib.dataog() / "ml1m" / "ml-1m"
    r = pd.read_csv(src / "ratings.dat", sep="::", engine="python", names=["SessionId", "ItemId", "Rating", "ts"])
    r = r.sort_values(["SessionId", "ts"], kind="stable")
    r["Time"] = pd.to_datetime(r.ts, unit="s").dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    r["Reward"] = 1.0
    tmp = out / "ml1m_formatted_sessions.csv"
    r[["SessionId", "ItemId", "Time", "Reward"]].to_csv(tmp)
    ds = SessionDataset(filepath_or_bytes=str(tmp), n_withheld=1, evolving=False)
    ds.load_and_split(TemporalSplit(test_frac=0.2, num_folds=3, filter_non_trained_test_items=True))
    ds.to_pickle(str(out / "ml1m.pickle"))
    tmp.unlink()
    m = pd.read_csv(src / "movies.dat", sep="::", engine="python", names=["ItemId", "t", "g"], encoding="latin-1")
    m["text"] = m.t + ". Genres: " + m.g.str.replace("|", ", ", regex=False)
    m[["ItemId", "text"]].to_csv(out / "ml1m_items.csv", index=False)
    return len(r), r.SessionId.nunique(), r.ItemId.nunique()


def describe(name, ds, smoke):
    tr = ds.get_train_data()
    lens = tr.groupby("SessionId").size()
    prompts, gt = ds.get_test_prompts(), ds.get_test_ground_truths()
    plen = np.array([len(v) for v in prompts.values()])
    pop = tr.ItemId.value_counts()
    stats = {"train_interactions": len(tr), "train_sessions": tr.SessionId.nunique(), "items": int(ds.get_unique_item_count()),
             "test_sessions": len(prompts), "train_len_mean": lens.mean(), "train_len_median": lens.median(),
             "test_prompt_len_mean": plen.mean(), "density": len(tr) / (tr.SessionId.nunique() * ds.get_unique_item_count())}
    run = lib.new_run(f"EDA-{name.upper()}-01", "mo-ta", "na", "thong-ke", smoke=smoke, unit=f"eda:{name}")
    log = lib.RunLog(run)
    # figures: session length, item popularity (long tail), activity over time
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(lens.clip(upper=lens.quantile(0.99)), bins=50, color="#4C72B0")
    ax.set_xlabel("so tuong tac / phien (train, cat o p99)"); ax.set_ylabel("so phien"); ax.set_title(f"{name}: do dai phien")
    fig.tight_layout(); fig.savefig(run / "figures" / f"fig_3-01_{name}_do-dai-phien.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.loglog(np.arange(1, len(pop) + 1), pop.values, color="#C44E52")
    ax.set_xlabel("hang do pho bien cua item"); ax.set_ylabel("so tuong tac"); ax.set_title(f"{name}: phan bo do pho bien (log-log)")
    fig.tight_layout(); fig.savefig(run / "figures" / f"fig_3-02_{name}_do-pho-bien.png", dpi=150); plt.close(fig)
    if "Time" in tr.columns:
        t = pd.to_datetime(tr.Time).dt.to_period("M").value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot(t.index.to_timestamp(), t.values, color="#55A868")
        ax.set_ylabel("so tuong tac / thang"); ax.set_title(f"{name}: hoat dong theo thoi gian (train)")
        fig.tight_layout(); fig.savefig(run / "figures" / f"fig_3-03_{name}_theo-thoi-gian.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(plen.clip(max=np.quantile(plen, 0.99)), bins=40, color="#8172B2")
    ax.set_xlabel("do dai prompt cua phien test"); ax.set_ylabel("so phien"); ax.set_title(f"{name}: phien test")
    fig.tight_layout(); fig.savefig(run / "figures" / f"fig_3-04_{name}_prompt-test.png", dpi=150); plt.close(fig)
    pd.Series(stats).to_csv(run / "data" / f"bang_3-01_{name}_mo-ta.csv", header=["gia_tri"])
    pop.head(50).to_csv(run / "data" / f"bang_3-02_{name}_top50-item.csv", header=["so_tuong_tac"])
    lib.write_config(run, {"dataset": name, "pickle": f"dataset/{name}.pickle"})
    log.event("eda", "xong")
    lib.write_metrics(run, [(k, v, "train" if k.startswith("train") or k in ("items", "density") else "test", "mo_ta", "EDA") for k, v in stats.items()])
    lib.log_book(smoke, f"EDA-{name.upper()}-01", name, "mo-ta", "na", run, "thong ke mo ta + hinh", True)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    out = lib.ds_dir(a.smoke)            # smoke never touches the real dataset/
    out.mkdir(exist_ok=True)
    t = time.time()
    run = lib.new_run("PRE-ALL-01", "chuan-bi", "temporal-split", "beauty-ml1m", smoke=a.smoke, unit="prep")
    log = lib.RunLog(run)
    prep_beauty(out)
    log.event("beauty", "chep pickle + 2 file embedding goc + items", sec=time.time() - t)
    n, u, i = prep_ml1m(out)
    log.event("ml1m", f"raw {n} tuong tac, {u} nguoi dung, {i} phim -> TemporalSplit", sec=time.time() - t)
    rows = []
    for name in lib.DATASETS:
        ds = SessionDataset.from_pickle(str(out / f"{name}.pickle"))
        s = describe(name, ds, a.smoke)
        rows += [(f"{name}_{k}", v, "dataset", name, "PRE") for k, v in s.items()]
    # before/after table: raw ML-1M vs what the split keeps for training
    pd.DataFrame({"truoc (ratings.dat)": [n, u, i], "sau (train cua TemporalSplit)": [s["train_interactions"], s["train_sessions"], s["items"]]},
                 index=["tuong_tac", "phien/nguoi_dung", "item"]).to_csv(run / "data" / "bang_3-03_ml1m_truoc-sau.csv")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar(["truoc", "sau (train)"], [n, s["train_interactions"]], color=["#999999", "#4C72B0"]); ax.set_title("ML-1M: so tuong tac truoc/sau chia")
    fig.tight_layout(); fig.savefig(run / "figures" / "fig_3-05_ml1m_truoc-sau.png", dpi=150); plt.close(fig)
    missing = [p.name for p in outputs(a.smoke) if not p.exists()]
    if missing:
        raise SystemExit(f"01 thieu dau ra: {missing}")
    lib.write_config(run, {"outputs": [p.name for p in outputs(a.smoke)], "split": "TemporalSplit(test_frac=0.2, num_folds=3, filter_non_trained_test_items=True)"})
    lib.write_metrics(run, rows)
    lib.log_book(a.smoke, "PRE-ALL-01", "beauty+ml1m", "chuan-bi", "temporal-split", run, "dataset/ tu DataOG/", True)
    print(f"[01] OK {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
