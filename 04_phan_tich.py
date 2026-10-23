"""04 — analysis of every finished REC run (reads Result/ only): run ANA-ALL-01 with
  bang_4-01_tong_hop.csv        mean ± std, n seeds, max-of-k of every metric per (dataset, model, embedding)
  bang_4-02_so_sanh_chinh.csv   pre-registered comparisons C1–C4 (Holm) + secondary ones (descriptive) — ho_so/09_exp_plan.md
  bang_4-03_khong_kem_hon.csv   H4 non-inferiority of open vs commercial embeddings (delta = 10 % of the commercial mean; decision row BGE vs PaLM)
  bang_4-04_lac_quan_bestofk.csv H2: max-of-k vs mean (the public notebook keeps the best of 5 runs by test NDCG@20)
  bang_4-05 absolute replication vs published numbers · bang_4-06 decisions H1-H6 · bang_4-07 initial-table statistics
and the figures. PRIMARY test = paired differences across runs (same seed), t CI, Holm m = 4 (Gusak RecSys'26);
per-session Wilcoxon + bootstrap on seed-averaged NDCG@20 is SECONDARY."""
import argparse, json

import numpy as np
import pandas as pd

import lib
import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = ["NDCG@20", "HitRate@20", "MRR@20", "NDCG@10", "HitRate@10", "MRR@10", "Catalog coverage@20", "train_sec", "epochs_run"]
FAMILY = {"random": "Non-neural", "llm2sasrec_shuf": "LLM-init", "popular": "Non-neural", "vsknn": "Non-neural", "gru4rec": "ID-only", "sasrec": "ID-only", "sasrec_l2s": "ID-only", "bert4rec": "ID-only",
          "llm2sasrec": "LLM-init", "llm2bert4rec": "LLM-init", "fused": "Fused (ours)", "llmseqsim": "LLMSeqSim"}
# dataviz reference palette (validated slots) + texture for B/W print
COLOR = {"Non-neural": "#898781", "ID-only": "#2a78d6", "LLM-init": "#eb6834", "Fused (ours)": "#1baf7a", "LLMSeqSim": "#6250d6"}
HATCH = {"Non-neural": "", "ID-only": "", "LLM-init": "//", "Fused (ours)": "xx", "LLMSeqSim": ".."}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
PRIMARY = {  # name: (dataset, (model, emb) a, (model, emb) b, hypothesis)
    "C1": ("beauty", ("llm2sasrec", "palm"), ("sasrec", "-"), "H1"),
    "C2": ("beauty", ("llm2sasrec", "palm"), ("sasrec_l2s", "-"), "H3"),
    "C3": ("ml1m", ("llm2sasrec", "bge-base"), ("sasrec_l2s", "-"), "H5"),
    "C4": ("beauty", ("fused", "bge-base"), ("llm2sasrec", "bge-base"), "H6"),
}
SECONDARY = [
    ("beauty", ("llm2sasrec", "bge-base"), ("sasrec_l2s", "-"), "H3/H4"), ("beauty", ("llm2sasrec", "bge-base"), ("sasrec", "-"), "H1/H4"),
    ("beauty", ("llm2sasrec", "minilm"), ("sasrec_l2s", "-"), "H4"), ("beauty", ("llm2sasrec", "openai"), ("sasrec_l2s", "-"), "H3/ada"), ("beauty", ("llm2sasrec", "openai"), ("sasrec", "-"), "H1/ada"),
    ("ml1m", ("llm2sasrec", "bge-base"), ("sasrec", "-"), "H5"), ("ml1m", ("llm2sasrec", "minilm"), ("sasrec_l2s", "-"), "H5"),
    ("ml1m", ("fused", "bge-base"), ("llm2sasrec", "bge-base"), "H6"), ("beauty", ("fused", "openai"), ("llm2sasrec", "openai"), "H6"),
    ("beauty", ("llmseqsim", "bge-base"), ("llmseqsim", "openai"), "H4"), ("beauty", ("sasrec_l2s", "-"), ("sasrec", "-"), "cau hinh"),
    ("ml1m", ("sasrec_l2s", "-"), ("sasrec", "-"), "cau hinh"), ("beauty", ("llm2bert4rec", "palm"), ("bert4rec", "-"), "H1-BERT"),
    ("ml1m", ("llm2bert4rec", "bge-base"), ("bert4rec", "-"), "H5-BERT"),
    ("beauty", ("llm2sasrec", "palm"), ("llm2sasrec_shuf", "palm"), "A1-ngu-nghia"), ("beauty", ("llm2sasrec_shuf", "palm"), ("sasrec_l2s", "-"), "A1-phan-bo"),
    ("beauty", ("popular", "-"), ("random", "-"), "R3"), ("ml1m", ("popular", "-"), ("random", "-"), "R3"),
]
# published numbers for the absolute-replication table (Amazon Beauty, NDCG@20): Boz et al. Table 4; Harte et al. Table 2
PUBLISHED = [("llm2sasrec", "palm", 0.054, "Boz 2025 Bang 4"), ("sasrec", "-", 0.033, "Boz 2025 Bang 4"), ("llm2bert4rec", "palm", 0.052, "Boz 2025 Bang 4"),
             ("bert4rec", "-", 0.041, "Boz 2025 Bang 4"), ("vsknn", "-", 0.043, "Boz 2025 Bang 4"), ("llmseqsim", "openai", 0.048, "Harte 2023 Bang 2")]
H4_DECISION = ("beauty", "llm2sasrec", "bge-base", "palm")
NONINF = [("beauty", "llm2sasrec", "bge-base", "palm"), ("beauty", "llm2sasrec", "minilm", "palm"), ("beauty", "llm2sasrec", "bge-base", "openai"),
          ("beauty", "llm2sasrec", "minilm", "openai"), ("beauty", "llmseqsim", "bge-base", "palm"), ("beauty", "llmseqsim", "minilm", "palm"),
          ("beauty", "llmseqsim", "bge-base", "openai"), ("beauty", "llmseqsim", "minilm", "openai"), ("beauty", "llmseqsim", "bge-small", "openai"),
          ("beauty", "fused", "bge-base", "openai")]


def label(model, emb):
    from models import NAMES
    return NAMES[model] + ("" if emb == "-" else f" ({emb})")


def collect(smoke):
    rows, ps = [], {}
    for d, cfg in lib.runs(smoke):
        if not str(cfg.get("unit", "")).startswith("rec:"):
            continue
        m = pd.read_csv(d / "metrics.csv").set_index("ten_do_do")["gia_tri"]
        rows.append({"dataset": cfg["dataset"], "model": cfg["model_key"], "embedding": cfg["embedding"], "seed": cfg["seed"], "run": d.name,
                     "machine": cfg.get("machine", "?"),
                     **{k: m.get(k, np.nan) for k in M}})
        p = d / "data" / "per_session.csv"
        if p.exists():
            ps.setdefault((cfg["dataset"], cfg["model_key"], cfg["embedding"]), []).append(pd.read_csv(p).set_index("SessionId")["ndcg20"])
    return pd.DataFrame(rows), {k: pd.concat(v, axis=1).mean(axis=1) for k, v in ps.items()}


def summary(df):
    g = df.groupby(["dataset", "model", "embedding"])
    out = g[M].mean().add_suffix("_mean").join(g[M].std(ddof=1).add_suffix("_std")).join(g.size().rename("n_seeds"))
    out = out.join(g["NDCG@20"].max().rename("NDCG@20_max")).reset_index()
    out["family"] = out.model.map(FAMILY)
    out["label"] = [label(m, e) for m, e in zip(out.model, out.embedding)]
    return out.sort_values(["dataset", "NDCG@20_mean"], ascending=[True, False])


def seeds_of(df, ds, m, e):
    return df[(df.dataset == ds) & (df.model == m) & (df.embedding == e)]


def compare(ps, df, spec_list, names=None):
    """PRIMARY: paired differences across runs (same seed for both sides; t CI over the seed pairs — Gusak RecSys'26).
    SECONDARY ("these trained models"): per-session Wilcoxon + bootstrap CI on seed-averaged NDCG@20.
    A pair whose two sides ran on different machines is flagged (one tier, one machine)."""
    rows = []
    for i, (ds, a, b, hyp) in enumerate(spec_list):
        ka, kb = (ds, *a), (ds, *b)
        rid = names[i] if names else f"S{i + 1}"
        if ka in ps and kb in ps:
            r = stats.paired(ps[ka], ps[kb])
            sa, sb = seeds_of(df, *ka), seeds_of(df, *kb)
            sp = stats.seed_paired(sa.set_index("seed")["NDCG@20"], sb.set_index("seed")["NDCG@20"])
            rows.append({"id": rid, "hypothesis": hyp, "dataset": ds, "a": label(*a), "b": label(*b), **sp,
                         "rel_diff_pct": 100 * sp["seed_mean_diff"] / sb["NDCG@20"].mean() if len(sb) and sb["NDCG@20"].mean() else np.nan,
                         **{f"phien_{k}": v for k, v in r.items()}, "n_seed_a": len(sa), "n_seed_b": len(sb),
                         "cung_may": len(set(sa.machine) | set(sb.machine)) == 1, "trang_thai": "du"})
        elif names:                                    # pre-registered comparison not available yet: listed, never dropped
            rows.append({"id": rid, "hypothesis": hyp, "dataset": ds, "a": label(*a), "b": label(*b), "trang_thai": "thieu", "seed_p": 1.0})
    return pd.DataFrame(rows)


def init_stats(smoke):
    """A1 (G2): what the initial item table looks like — row L2 norms and variance share of the top PCA axes for the PCA(352)
    initialisation of every embedding source vs the Keras default random init (uniform(-0.05, 0.05), 352 dims)."""
    import json
    from sklearn.decomposition import PCA
    out, rng = [], np.random.default_rng(42)
    for ds in lib.DATASETS:
        R = rng.uniform(-0.05, 0.05, (5000, 352))
        n = np.linalg.norm(R, axis=1)
        out.append({"dataset": ds, "nguon": "random (Keras uniform +-0.05)", "dim": 352, "norm_mean": n.mean(), "norm_std": n.std(), "var_top1": np.nan, "var_top10": np.nan})
        for e in ("palm", "openai", "bge-base", "minilm"):
            p = lib.ds_dir(smoke) / f"{ds}_emb_{e}.csv.gz"
            if not p.exists():
                continue
            E = np.vstack([np.asarray(json.loads(x), dtype=np.float32) for x in pd.read_csv(p).embedding])
            k = min(352, E.shape[1])
            pca = PCA(n_components=k).fit(E)
            Z = pca.transform(E)
            n = np.linalg.norm(Z, axis=1)
            out.append({"dataset": ds, "nguon": f"PCA({k}) {e}", "dim": k, "norm_mean": n.mean(), "norm_std": n.std(),
                        "var_top1": pca.explained_variance_ratio_[0] / pca.explained_variance_ratio_.sum(),
                        "var_top10": pca.explained_variance_ratio_[:10].sum() / pca.explained_variance_ratio_.sum()})
    return pd.DataFrame(out)


def bars(sm, ds, path):
    s = sm[sm.dataset == ds].sort_values("NDCG@20_mean")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(s) + 1.2))
    y = np.arange(len(s))
    for yi, (_, r) in zip(y, s.iterrows()):
        ax.barh(yi, r["NDCG@20_mean"], xerr=r["NDCG@20_std"] if r["n_seeds"] > 1 else None, color=COLOR[r.family], hatch=HATCH[r.family],
                edgecolor="white", height=0.7, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
    ax.set_yticks(y, s.label, fontsize=8, color=INK)
    ax.set_xlabel("NDCG@20 (mean ± std over seeds)", color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[f], hatch=HATCH[f], ec="white") for f in COLOR if f in set(s.family)]
    ax.legend(handles, [f for f in COLOR if f in set(s.family)], fontsize=7, frameon=False, loc="lower right")
    ax.set_title({"beauty": "Amazon Beauty", "ml1m": "MovieLens-1M"}[ds], color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def seeds_plot(df, ds, path):
    s = df[(df.dataset == ds)]
    keys = s.groupby(["model", "embedding"])["NDCG@20"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(keys) + 1.2))
    for i, (m, e) in enumerate(keys):
        v = s[(s.model == m) & (s.embedding == e)]["NDCG@20"].values
        ax.scatter(v, np.full(len(v), i), s=18, color=COLOR[FAMILY[m]], edgecolor="white", linewidth=0.8, zorder=3)
        ax.plot([v.min(), v.max()], [i, i], color=MUTED, lw=0.8, zorder=2)
    ax.set_yticks(range(len(keys)), [label(m, e) for m, e in keys], fontsize=8)
    ax.set_xlabel("NDCG@20 of each seed", color=MUTED); ax.grid(axis="x", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"{ {'beauty': 'Amazon Beauty', 'ml1m': 'MovieLens-1M'}[ds]}: run-to-run spread", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def forest(cmp, path):
    if cmp.empty:
        return
    c = cmp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.36 * len(c) + 1.2))
    y = np.arange(len(c))
    ax.errorbar(c.seed_mean_diff, y, xerr=[c.seed_mean_diff - c.seed_ci_lo, c.seed_ci_hi - c.seed_mean_diff], fmt="o", color="#2a78d6", ecolor=INK, elinewidth=1, capsize=2, ms=5)
    ax.axvline(0, color=MUTED, lw=0.8, ls="--")
    ax.set_yticks(y, [f"{r.dataset}: {r.a} − {r.b}" for r in c.itertuples()], fontsize=7)
    ax.set_xlabel("Δ NDCG@20, paired over seeds (95 % t-CI)", color=MUTED); ax.grid(axis="x", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def protocol_plot(h2, path):
    if h2.empty:
        return
    h = h2.sort_values("mean")
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(h) + 1.2))
    y = np.arange(len(h))
    ax.scatter(h["mean"], y, color="#2a78d6", s=22, label="mean of seeds (this study)", zorder=3)
    ax.scatter(h["max"], y, color="#eb6834", marker="D", s=22, label="best of seeds by test NDCG@20 (original protocol)", zorder=3)
    for yi, (lo, hi) in zip(y, zip(h["mean"], h["max"])):
        ax.plot([lo, hi], [yi, yi], color=MUTED, lw=0.8)
    ax.set_yticks(y, [f"{r.dataset}: {r.label}" for r in h.itertuples()], fontsize=7)
    ax.set_xlabel("NDCG@20", color=MUTED); ax.legend(fontsize=7, frameon=False, loc="lower right"); ax.grid(axis="x", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def source_plot(sm, path):
    s = sm[(sm.dataset == "beauty") & sm.model.isin(["llm2sasrec", "llmseqsim", "fused"])]
    if s.empty:
        return
    embs = [e for e in ["openai", "palm", "bge-base", "bge-small", "minilm"] if e in set(s.embedding)]
    mods = [m for m in ["llm2sasrec", "fused", "llmseqsim"] if m in set(s.model)]
    w = 0.8 / len(embs)
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax, seen = *plt.subplots(figsize=(6.4, 3.2)), set()
    for j, e in enumerate(embs):
        for i, m in enumerate(mods):
            r = s[(s.model == m) & (s.embedding == e)]
            if len(r):
                ax.bar(i + (j - (len(embs) - 1) / 2) * w, r["NDCG@20_mean"].iloc[0], w * 0.92, color=pal[j], edgecolor="white",
                       yerr=r["NDCG@20_std"].iloc[0] if r["n_seeds"].iloc[0] > 1 else None, error_kw={"elinewidth": 1, "capsize": 2},
                       label=None if e in seen else e)
                seen.add(e)
    from models import NAMES
    ax.set_xticks(range(len(mods)), [NAMES[m] for m in mods]); ax.set_ylabel("NDCG@20", color=MUTED)
    ax.legend(title="embedding", fontsize=7, title_fontsize=7, frameon=False, ncol=len(embs)); ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Amazon Beauty: commercial vs open-source text embeddings", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    df, ps = collect(a.smoke)
    df = df.drop_duplicates(["dataset", "model", "embedding", "seed"], keep="first")   # Result/ merged from two machines
    keep = set(df.run)                                                                  # per-session table only from the kept runs
    ps = {k: pd.concat([pd.read_csv(lib.res_dir(a.smoke) / r / "data" / "per_session.csv").set_index("SessionId")["ndcg20"]
                        for r in df[(df.dataset == k[0]) & (df.model == k[1]) & (df.embedding == k[2])].run
                        if (lib.res_dir(a.smoke) / r / "data" / "per_session.csv").exists()], axis=1).mean(axis=1)
          for k in ps if any(r in keep for r in df[(df.dataset == k[0]) & (df.model == k[1]) & (df.embedding == k[2])].run)}
    run = lib.new_run("ANA-ALL-01", "phan-tich", "holm-bootstrap", "tong-hop", smoke=a.smoke, unit="ana")
    log = lib.RunLog(run)
    if df.empty:
        raise SystemExit("04: chua co run REC nao co metrics.csv")
    D, F = run / "data", run / "figures"
    df.to_csv(D / "bang_4-00_tung_run.csv", index=False)
    sm = summary(df)
    assert len(sm) == df.groupby(["dataset", "model", "embedding"]).ngroups  # one row per configuration ([BH-017])
    sm.to_csv(D / "bang_4-01_tong_hop.csv", index=False)
    prim = compare(ps, df, list(PRIMARY.values()), list(PRIMARY))
    # Holm over the FOUR pre-registered comparisons on the RUN-LEVEL p-values (a missing one counts as p = 1).
    # Verdict = Holm-p < 0.05 AND seed-pair CI excludes 0 (09, after G1.5/G2); otherwise "khong ket luan duoc" (5 seeds, low power)
    adj = stats.holm(dict(zip(prim.id, prim.seed_p.fillna(1.0))))
    prim["holm_p"] = prim.id.map(adj)
    ok = (prim.trang_thai == "du") & (prim.holm_p < 0.05) & ((prim.seed_ci_lo > 0) | (prim.seed_ci_hi < 0)) & (prim.cung_may == True)
    prim["verdict"] = np.where(prim.trang_thai != "du", "thieu", np.where(ok, np.where(prim.seed_mean_diff > 0, "a > b", "a < b"), "khong ket luan duoc"))
    sec = compare(ps, df, SECONDARY)
    if not sec.empty:
        sec["holm_p"] = np.nan
        sec["verdict"] = np.where((sec.seed_p < 0.05) & ((sec.seed_ci_lo > 0) | (sec.seed_ci_hi < 0)), np.where(sec.seed_mean_diff > 0, "a > b (mo ta)", "a < b (mo ta)"), "khong ket luan duoc (mo ta)")
    cmp = pd.concat([prim.assign(family="chinh"), sec.assign(family="phu")], ignore_index=True)
    cmp.to_csv(D / "bang_4-02_so_sanh_chinh.csv", index=False)
    ni = []
    for ds, m, e, ref in NONINF:
        ka, kb = (ds, m, e), (ds, m, ref)
        if ka in ps and kb in ps:
            sa, sb = seeds_of(df, *ka), seeds_of(df, *kb)
            sp = stats.seed_paired(sa.set_index("seed")["NDCG@20"], sb.set_index("seed")["NDCG@20"])
            delta = 0.10 * sb["NDCG@20"].mean()
            r = stats.paired(ps[ka], ps[kb])
            ni.append({"dataset": ds, "model": m, "open": e, "commercial": ref, **sp, "delta": delta,
                       "non_inferior": bool(sp["seed_ci_lo"] > -delta) if sp["seed_ci_lo"] == sp["seed_ci_lo"] else None,
                       "phien_ci_lo": r["ci_lo"], "phien_non_inferior": r["ci_lo"] > -delta, "quyet_dinh_H4": (ds, m, e, ref) == H4_DECISION})
    pd.DataFrame(ni).to_csv(D / "bang_4-03_khong_kem_hon.csv", index=False)
    c1 = prim.loc[prim.id == "C1", "seed_mean_diff"].iloc[0] if (not prim.empty and "C1" in set(prim.id) and "seed_mean_diff" in prim) else np.nan
    h2 = []
    for (ds, m, e), g in df.sort_values("seed").groupby(["dataset", "model", "embedding"]):
        if len(g) >= 3:
            v = g["NDCG@20"].values
            h2.append({"dataset": ds, "label": label(m, e), "n": len(v), "mean": v.mean(), "max": v.max(), "max_minus_mean": v.max() - v.mean(),
                       "max3_minus_mean3": v[:3].max() - v[:3].mean(), "rel_to_C1_pct": 100 * (v.max() - v.mean()) / c1 if c1 == c1 and c1 else np.nan,
                       "lac_quan_dang_ke": bool(c1 == c1 and c1 and (v.max() - v.mean()) >= 0.10 * abs(c1))})
    h2 = pd.DataFrame(h2)
    h2.to_csv(D / "bang_4-04_lac_quan_bestofk.csv", index=False)
    # H2 decision statistic (G2): inflation of the C1 gap = (max5(LLM2SASRec-PaLM) - max5(SASRec)) - C1
    a1, b1 = seeds_of(df, "beauty", "llm2sasrec", "palm")["NDCG@20"], seeds_of(df, "beauty", "sasrec", "-")["NDCG@20"]
    dmax = a1.max() - b1.max() if len(a1) and len(b1) else np.nan
    h2_infl = dmax - c1 if dmax == dmax and c1 == c1 else np.nan
    # absolute replication (G2): published number vs our min / mean / max over seeds
    rep = []
    for m, e, pub, src in PUBLISHED:
        v = seeds_of(df, "beauty", m, e)["NDCG@20"]
        if len(v):
            rep.append({"model": label(m, e), "nguon": src, "cong_bo": pub, "n_seed": len(v), "min": v.min(), "mean": v.mean(), "max": v.max(),
                        "tai_lap_duoc": bool(v.min() <= pub <= v.max() or abs(v.max() - pub) <= 0.10 * pub)})
    pd.DataFrame(rep).to_csv(D / "bang_4-05_tai_lap_so.csv", index=False)
    init_stats(a.smoke).to_csv(D / "bang_4-07_khoi_tao.csv", index=False)
    # decisions H1-H6 exactly as pre-registered (09_exp_plan.md, locked)
    get = lambda t, i: (t.loc[t.id == i].iloc[0] if (not t.empty and i in set(t.id)) else None)
    def sec_row(a_lbl, b_lbl, ds):
        if sec.empty:
            return None
        s = sec[(sec.a == a_lbl) & (sec.b == b_lbl) & (sec.dataset == ds)]
        return s.iloc[0] if len(s) else None
    c3, s6 = get(prim, "C3"), sec_row(label("llm2sasrec", "bge-base"), label("sasrec", "-"), "ml1m")
    h4 = next((x for x in ni if x["quyet_dinh_H4"]), None)
    dec = [{"gia_thuyet": "H1", "quy_tac": "C1 a > b (t ghep cap theo seed: Holm m=4 + CI)", "ket_qua": get(prim, "C1")["verdict"] if get(prim, "C1") is not None else "thieu"},
           {"gia_thuyet": "H3", "quy_tac": "C2: a > b => DAT; a < b => do cau hinh; khong ket luan duoc => khong ket luan duoc (doc kem epochs_run, cau hinh L2S patience 2)",
            "ket_qua": "thieu" if get(prim, "C2") is None else {"a > b": "DAT", "a < b": "loi ich chu yeu do cau hinh"}.get(get(prim, "C2")["verdict"], get(prim, "C2")["verdict"])},
           {"gia_thuyet": "H2", "quy_tac": "lam phong = (max5 L2S-PaLM - max5 SASRec) - C1 >= 10 % C1 => lac quan dang ke", "gia_tri": h2_infl,
            "ket_qua": "thieu" if h2_infl != h2_infl else ("lac quan dang ke" if h2_infl >= 0.10 * abs(c1) else "khong dang ke (H2 bi bac)")},
           {"gia_thuyet": "H4", "quy_tac": "LLM2SASRec BGE-base vs PaLM, Beauty: can duoi CI cap-seed > -delta", "ket_qua": "thieu" if (h4 is None or h4["non_inferior"] is None) else ("khong kem hon" if h4["non_inferior"] else "khong chung minh duoc khong-kem-hon")},
           {"gia_thuyet": "H5", "quy_tac": "chuyen duoc khi C3 a > b VA (ML-1M LLM2SASRec-BGE - SASRec goc) co seed_ci_hi >= 0",
            "ket_qua": "thieu" if (c3 is None or s6 is None or c3["verdict"] == "thieu") else (
                "chuyen duoc sang ML-1M" if (c3["verdict"] == "a > b" and s6["seed_ci_hi"] >= 0) else
                "khoi tao bu phan thieu huan luyen cua cau hinh L2S; cau hinh Beauty khong chuyen sang ML-1M" if c3["verdict"] == "a > b" else
                "khong chuyen duoc" if c3["verdict"] == "a < b" else "khong ket luan duoc (5 seed)")},
           {"gia_thuyet": "H6", "quy_tac": "C4 hai phia (t ghep cap theo seed: Holm m=4 + CI)", "ket_qua": get(prim, "C4")["verdict"] if get(prim, "C4") is not None else "thieu"}]
    pd.DataFrame(dec).to_csv(D / "bang_4-06_quyet_dinh.csv", index=False)
    for ds in sorted(set(df.dataset)):
        bars(sm, ds, F / f"fig_4-03_{ds}_ndcg20.png")
        seeds_plot(df, ds, F / f"fig_4-04_{ds}_tung-seed.png")
    forest(cmp[cmp.trang_thai == "du"], F / "fig_4-05_so-sanh-ghep-cap.png")
    protocol_plot(h2, F / "fig_4-06_bestofk-vs-mean.png")
    source_plot(sm, F / "fig_4-07_beauty_nguon-embedding.png")
    log.event("phan_tich", f"{len(df)} run, {len(sm)} cau hinh, {len(cmp)} so sanh, {len(ni)} khong-kem-hon, {len(h2)} cau hinh nhieu seed")
    lib.write_config(run, {"inputs": sorted(df.run), "n_runs": len(df), "primary": {k: list(map(str, v)) for k, v in PRIMARY.items()},
                           "noninferiority_delta": "0.10 x mean NDCG@20 of the commercial reference (PaLM for the H4 decision row)",
                           "primary_test": "paired differences across runs (same seed), t CI, Holm m=4 on C1-C4 (Gusak RecSys'26)",
                           "secondary_test": "per-session Wilcoxon + bootstrap CI (10000, seed 42) on seed-averaged NDCG@20"})
    rows = [(f"{r.dataset}|{r.label}|NDCG@20_mean", r["NDCG@20_mean"], "test", "next_item", r.label) for _, r in sm.iterrows()]
    rows += [(f"{r.id}|seed_mean_diff", r.seed_mean_diff, "test", r.dataset, f"{r.a} - {r.b}") for r in cmp.itertuples() if r.trang_thai == "du"]
    lib.write_metrics(run, rows)
    lib.log_book(a.smoke, "ANA-ALL-01", "beauty+ml1m", "phan-tich", "holm-bootstrap", run, "tong hop + kiem dinh + hinh", True)
    print(f"[04] {len(df)} run, {len(sm)} cau hinh, {len(cmp)} so sanh -> {run.name}", flush=True)


if __name__ == "__main__":
    main()
