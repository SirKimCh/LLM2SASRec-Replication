import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lib


def test_corrupt_state(tmp_path):
    p = tmp_path / "run_state.json"
    p.write_text('{"done": ', encoding="utf-8")          # torn write (power cut)
    assert lib.load_state(p) == {"done": {}}
    lib.save_state(p, {"done": {"a": 1}})
    assert lib.load_state(p) == {"done": {"a": 1}}


def test_unit_done_requires_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "ROOT", tmp_path)
    run = lib.new_run("REC-ML1M-01", "SASRec", "s42", "x", smoke=False, unit="rec:ml1m:sasrec:-:42")
    assert lib.find_run("rec:ml1m:sasrec:-:42", smoke=False) is None   # no metrics.csv yet -> unfinished
    lib.write_metrics(run, [("NDCG@20", 0.1, "test", "next_item", "SASRec")])
    assert lib.find_run("rec:ml1m:sasrec:-:42", smoke=False) == run
    assert lib.find_run("rec:ml1m:sasrec:-:43", smoke=False) is None


def test_run_has_five_components(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "ROOT", tmp_path)
    run = lib.new_run("REC-ML1M-01", "SASRec", "s42", "x", smoke=False, unit="u")
    lib.write_config(run, {"a": 1})
    lib.RunLog(run).event("train", "ok")
    lib.write_metrics(run, [("NDCG@20", 0.1, "test", "next_item", "SASRec")])
    assert {p.name for p in run.iterdir()} == {"config.json", "log.csv", "metrics.csv", "figures", "data"}
    assert json.loads((run / "config.json").read_text(encoding="utf-8"))["unit"] == "u"


def test_paths_at_drive_root(monkeypatch):
    monkeypatch.setattr(lib, "ROOT", Path("C:/src"))
    for p in (lib.res_dir(False), lib.ds_dir(True), lib.dataog()):
        assert ".." not in p.parts and p.parts[:2] == Path("C:/src").parts
