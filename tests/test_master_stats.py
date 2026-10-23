import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
import stats

spec = importlib.util.spec_from_file_location("master", SRC / "99_master_run.py")
master = importlib.util.module_from_spec(spec)
spec.loader.exec_module(master)


def test_holm():
    out = stats.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert np.allclose([out["a"], out["b"], out["c"]], [0.03, 0.06, 0.06])


def test_bootstrap_ci_contains_mean():
    d = np.random.default_rng(0).normal(0.01, 0.05, 2000)
    lo, hi = stats.bootstrap_ci(d)
    assert lo < d.mean() < hi and hi - lo < 0.01


def test_paired_detects_difference():
    a = pd.Series(np.r_[np.ones(300), np.zeros(700)], index=range(1000))
    b = pd.Series(np.zeros(1000), index=range(1000))
    r = stats.paired(a, b)
    assert r["mean_diff"] == 0.3 and r["wilcoxon_p"] < 1e-6 and r["n"] == 1000


def test_units_order_tiers():
    us = master.units("C")
    keys = [u.key for u in us]
    assert keys[:2] == ["env", "prep"] and keys[-2:] == ["ana", "web"]
    tiers = [u.tier for u in us if u.tier in "ABC"]
    assert tiers == sorted(tiers)                                  # A before B before C
    assert len([u for u in master.units("A") if u.kind == "rec"]) == 4 + 7 * 5 + 3 + 3 * 5
    for u in us:                                                   # every embedding a model needs is produced in its tier or earlier
        if u.kind == "rec" and u.cmd[6] not in ("-", "openai", "palm"):
            emb = f"emb:{u.cmd[2]}:{u.cmd[6]}"
            assert any(v.key == emb for v in us[: us.index(u)]), (u.key, emb)


def test_other_run_refused(tmp_path):
    procs = [(10, ["python", "99_master_run.py"], str(tmp_path)), (11, ["python", "99_master_run.py", "--estimate"], str(tmp_path)),
             (12, ["python", "99_master_run.py"], str(tmp_path / "other"))]
    assert master.other_run(procs, {1}, tmp_path) == 10
    assert master.other_run(procs[1:], {1}, tmp_path) is None


def test_clean_bat_fake_dir(tmp_path):
    for d in (".venv", "DataOG", "dataset", "Result", ".cache"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "x.txt").write_text("x")
    for f in ("99_master_run.py", "a.py", "hardware_profile.json"):
        (tmp_path / f).write_text("x")
    shutil.copy(SRC / "clean.bat", tmp_path / "clean.bat")
    r = subprocess.run(["cmd", "/c", str(tmp_path / "clean.bat"), "--yes"], cwd=tmp_path, input="\n", capture_output=True, text=True)
    left = {p.name for p in tmp_path.iterdir()}
    assert r.returncode == 0, r.stdout + r.stderr
    assert left == {"Result", "99_master_run.py", "a.py", "clean.bat"}, left


def test_welch_seed_level():
    assert stats.welch([0.05, 0.051, 0.049, 0.05, 0.052], [0.03, 0.031, 0.029, 0.03, 0.032]) < 1e-4
    assert stats.welch([0.05, 0.051, 0.049], [0.05, 0.049, 0.051]) > 0.5
    assert np.isnan(stats.welch([0.05], [0.04, 0.03]))


def test_random_baseline_excludes_prompt_and_is_seeded():
    import models
    m = models.RandomRec(seed=7)
    m.train(pd.DataFrame({"ItemId": np.arange(100)}))
    prompts = {1: np.array([0, 1, 2]), 2: np.array([5])}
    p1, p2 = m.predict(prompts, top_k=20), models.RandomRec(seed=7)
    p2.train(pd.DataFrame({"ItemId": np.arange(100)}))
    assert all(len(v) == 20 and not np.isin(v, prompts[k]).any() for k, v in p1.items())
    assert all((p1[k] == v).all() for k, v in p2.predict(prompts, top_k=20).items())


def test_seed_paired():
    a = pd.Series([0.050, 0.052, 0.049, 0.051, 0.050], index=[42, 43, 44, 45, 46])
    b = pd.Series([0.030, 0.031, 0.029, 0.033, 0.030], index=[42, 43, 44, 45, 46])
    r = stats.seed_paired(a, b)
    assert r["n_pairs"] == 5 and abs(r["seed_mean_diff"] - 0.0198) < 1e-9 and r["seed_ci_lo"] > 0 and r["seed_p"] < 1e-4
    r = stats.seed_paired(pd.Series([0.05, 0.04], index=[42, 43]), pd.Series([0.04, 0.05], index=[42, 43]))
    assert r["seed_ci_lo"] < 0 < r["seed_ci_hi"]
