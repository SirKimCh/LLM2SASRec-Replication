"""Shared helpers: paths, seeds, run folders (05 §3–§5), the experiment log book, resume state.
Every path is relative to this folder (src/), so src/ can be copied anywhere (also C:\\src)."""
import csv, getpass, json, os, platform, random, re, sys, time
from datetime import datetime
from pathlib import Path

# A machine-wide HF_HUB_ENABLE_HF_TRANSFER (hf_transfer is not in uv.lock) or an *_OFFLINE flag would break the
# downloads of 00/02 on other machines ([BH-022]): drop them before any Hugging Face library is imported.
for _v in ("HF_HUB_ENABLE_HF_TRANSFER", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.pop(_v, None)

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor" / "llmseqrec"          # original code of Harte et al. / Boz et al. (BSD-3), unchanged
sys.path.insert(0, str(VENDOR))
VENDOR_COMMIT = "6464b39064f369b251580c63edbf635de0934101"

DATASETS = ("beauty", "ml1m")
BOOK_NOTE = ("# Ma TN: ENV=moi truong, PRE=chuan bi du lieu, EDA=thong ke mo ta, EMB=sinh embedding van ban, "
             "REC=huan luyen/danh gia mo hinh goi y, ANA=phan tich tong hop, WEB=xuat du lieu ung dung web")
BOOK_COLS = ["ma_tn", "thoi_gian", "nguoi", "dataset", "mo_hinh", "sieu_tham_so", "thu_muc_result", "mo_ta", "trang_thai"]


def dataog():
    return ROOT / "DataOG"


def ds_dir(smoke=False):
    return ROOT / ("dataset_smoke" if smoke else "dataset")


def res_dir(smoke=False):
    return ROOT / ("Result_smoke" if smoke else "Result")


def who():
    return f"{getpass.getuser()}@{platform.node()}"


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg, smoke=False):
    """Print and append to Result/master_log.txt (the run's timeline, also on crashes)."""
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    d = res_dir(smoke)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "master_log.txt", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        sys.modules["tensorflow"].random.set_seed(seed)


def _safe(s):
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", str(s)).strip("-") or "x"


def new_run(ma_tn, model, hp, note, smoke=False, unit=""):
    """Result/<ddMMyyyy_HH-MM_MaTN_Model_HP_Note>/ with figures/ and data/; config.json records the unit key."""
    stamp = datetime.now().strftime("%d%m%Y_%H-%M")
    base = res_dir(smoke) / "_".join([stamp, _safe(ma_tn), _safe(model), _safe(hp), _safe(note)])
    run, k = base, 2
    while run.exists():
        run, k = Path(f"{base}-{k}"), k + 1
    (run / "figures").mkdir(parents=True)
    (run / "data").mkdir()
    write_config(run, {"unit": unit, "ma_tn": ma_tn, "machine": who(), "started": now()})
    return run


def _atomic_write(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_config(run, cfg):
    p = run / "config.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    _atomic_write(p, json.dumps({**old, **cfg}, indent=1, ensure_ascii=False, default=str))


class RunLog:
    def __init__(self, run):
        self.p = run / "log.csv"
        if not self.p.exists():
            self.p.write_text("timestamp,buoc,noi_dung,trang_thai,thoi_gian_giay\n", encoding="utf-8")

    def event(self, buoc, noi_dung, ok=True, sec=0.0):
        with open(self.p, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([now(), buoc, noi_dung, "OK" if ok else "FAIL", round(sec, 2)])


def write_metrics(run, rows):
    """rows: (ten_do_do, gia_tri, tap, target, mo_hinh). Written last and atomically: its presence marks a finished run."""
    lines = ["ten_do_do,gia_tri,tap,target,mo_hinh"] + [f"{a},{float(b):.10g},{c},{d},{e}" for a, b, c, d, e in rows]
    _atomic_write(run / "metrics.csv", "\n".join(lines) + "\n")


def log_book(smoke, ma_tn, dataset, mo_hinh, sieu_tham_so, run, mo_ta, ok=True):
    p = res_dir(smoke) / "ThucNghiem_Log.csv"
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as f:
        if new:
            f.write(BOOK_NOTE + "\n")
            csv.writer(f).writerow(BOOK_COLS)
        csv.writer(f).writerow([ma_tn, now(), who(), dataset, mo_hinh, sieu_tham_so, Path(run).name, mo_ta, "OK" if ok else "FAIL"])


def runs(smoke=False):
    """Finished runs: folders with metrics.csv + config.json, as (path, config)."""
    out = []
    for d in sorted(res_dir(smoke).glob("*/")):
        if (d / "metrics.csv").exists() and (d / "config.json").exists():
            out.append((d, json.loads((d / "config.json").read_text(encoding="utf-8"))))
    return out


def find_run(unit, smoke=False):
    for d, cfg in runs(smoke):
        if cfg.get("unit") == unit:
            return d
    return None


def load_state(path, tell=print):
    try:
        st = json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}
        if not isinstance(st, dict):
            raise ValueError
    except ValueError:
        tell("!! run_state.json hong (bi ngat khi dang ghi?) -> bo qua; viec da xong duoc nhan ra tu Result/")
        st = {}
    st.setdefault("done", {})
    return st


def save_state(path, st):
    _atomic_write(Path(path), json.dumps(st, indent=1, ensure_ascii=False))


def ram_free_gb():
    import psutil
    return psutil.virtual_memory().available / 1e9


def plan():
    try:
        return json.loads((ROOT / "resource_plan.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"cores": 4}
