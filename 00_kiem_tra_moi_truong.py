"""00 — check the environment and the data on EVERY launch; download what is missing (05 §7, §10).
(1) cache variables point inside the project (never C:\\ unless src/.cache), (2) every library imports, (3) DataOG/ matches
data_manifest.json (size + MD5) — missing/corrupt files are downloaded, a failed download stops with a clear message,
(4) hardware_profile.json + resource_plan.json, (5) run ENV-ALL-01 with its five components."""
import argparse, hashlib, importlib, json, os, platform, shutil, sys, time, urllib.request, zipfile
from pathlib import Path

import lib

CACHE_VARS = ["UV_CACHE_DIR", "HF_HOME", "FASTEMBED_CACHE_PATH", "MPLCONFIGDIR"]
LIBS = ["tensorflow", "pandas", "numpy", "scipy", "sklearn", "numba", "multiprocess", "tenacity", "jinja2", "fastembed", "onnxruntime", "matplotlib", "tqdm", "psutil"]


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def download(url, dest, size):
    """Download to <dest>.part with a progress bar, then rename; 3 attempts. Raises on failure (no fallback)."""
    from tqdm import tqdm
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as f, \
                    tqdm(total=size, unit="B", unit_scale=True, desc=dest.name, mininterval=1, disable=not sys.stderr.isatty()) as bar:
                for b in iter(lambda: r.read(1 << 20), b""):
                    f.write(b)
                    bar.update(len(b))
            if part.stat().st_size != size:
                raise IOError(f"size {part.stat().st_size} != {size}")
            os.replace(part, dest)
            return
        except Exception as e:  # network errors: retry, then stop loudly
            print(f"  tai {url} lan {attempt}/3 loi: {e}", flush=True)
            time.sleep(5 * attempt)
    raise SystemExit(f"KHONG TAI DUOC {url} -> {dest}. Kiem tra mang roi chay lai (khong dung du lieu thay the).")


def check_data(log):
    man = json.loads((lib.ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for name, d in man["datasets"].items():
        files = d["files"]
        bad = [f for f in files if not (lib.dataog() / f["path"]).exists() or (lib.dataog() / f["path"]).stat().st_size != f["size"]
               or md5(lib.dataog() / f["path"]) != f["md5"]]
        if bad and "zip" in d:
            z = d["zip"]
            zp = lib.dataog() / z["into"] / Path(z["url"]).name
            print(f"[00] {name}: thieu/hong {len(bad)} file -> tai {z['url']}", flush=True)
            download(z["url"], zp, z["size"])
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(lib.dataog() / z["into"])
            zp.unlink()
        else:
            for f in bad:
                print(f"[00] {name}: thieu/hong {f['path']} -> tai", flush=True)
                download(f["url"], lib.dataog() / f["path"], f["size"])
        for f in files:
            p = lib.dataog() / f["path"]
            ok = p.exists() and p.stat().st_size == f["size"] and md5(p) == f["md5"]
            rows.append((name, f["path"], f["size"], f["md5"], ok))
            if not ok:
                raise SystemExit(f"DU LIEU SAI sau khi tai: {p} (size/MD5 khong khop manifest)")
        log.event("data", f"{name}: {len(files)} file khop MD5")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    run = lib.new_run("ENV-ALL-01", "env", "na", "kiem-tra", smoke=a.smoke, unit=f"env:{time.time():.0f}")
    log = lib.RunLog(run)
    # (1) cache variables
    for v in CACHE_VARS:
        val = os.environ.get(v, "")
        inside = val and (str(lib.ROOT) in val or Path(val).drive.upper() != "C:")
        if not inside:
            raise SystemExit(f"Bien {v}={val!r} chua tro vao cache cua du an (khong duoc cai/tai vao o C). Hay chay qua run.bat.")
    log.event("cache", ", ".join(f"{v}={os.environ[v]}" for v in CACHE_VARS))
    # (2) libraries
    vers = {}
    for m in LIBS:
        vers[m] = getattr(importlib.import_module(m), "__version__", "?")
    log.event("libs", json.dumps(vers))
    # (3) data
    t = time.time()
    rows = check_data(log)
    # (4) hardware + resource plan (TF 2.13 on Windows has no GPU support -> CPU)
    import psutil
    phys = psutil.cpu_count(logical=False) or 2
    hw = {"os": platform.platform(), "python": sys.version.split()[0], "cpu": platform.processor(), "cores_physical": phys,
          "cores_logical": psutil.cpu_count(), "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
          "ram_free_gb": round(lib.ram_free_gb(), 1), "disk_free_gb": round(shutil.disk_usage(lib.ROOT).free / 1e9, 1)}
    plan = {"device": "cpu", "cores": max(1, min(phys - 1, 4)), "note": "TensorFlow 2.13 on Windows runs on CPU; one unit at a time"}
    (lib.ROOT / "hardware_profile.json").write_text(json.dumps(hw, indent=1), encoding="utf-8")
    (lib.ROOT / "resource_plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8")
    # (5) run record
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(["RAM tong", "RAM trong", "Dia trong"], [hw["ram_total_gb"], hw["ram_free_gb"], hw["disk_free_gb"]], color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_ylabel("GB"); ax.set_title("Tai nguyen may luc chay")
    fig.tight_layout(); fig.savefig(run / "figures" / "fig_0-01_ENV_tai-nguyen.png", dpi=150); plt.close(fig)
    with open(run / "data" / "data_manifest.csv", "w", encoding="utf-8") as f:
        f.write("dataset,file,size,md5,ok\n" + "".join(f"{a},{b},{c},{d},{e}\n" for a, b, c, d, e in rows))
    lib.write_config(run, {"hardware": hw, "resource_plan": plan, "libraries": vers, "vendor_commit": lib.VENDOR_COMMIT})
    lib.write_metrics(run, [("ram_free_gb", hw["ram_free_gb"], "env", "may", "env"), ("disk_free_gb", hw["disk_free_gb"], "env", "may", "env"),
                            ("data_files_ok", sum(r[4] for r in rows), "env", "DataOG", "env"), ("check_sec", time.time() - t, "env", "DataOG", "env")])
    lib.log_book(a.smoke, "ENV-ALL-01", "beauty+ml1m", "env", "na", run, "kiem tra moi truong + du lieu", True)
    print(f"[00] OK — {len(rows)} file du lieu khop MD5, {plan['cores']} nhan cho TF, RAM trong {hw['ram_free_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
