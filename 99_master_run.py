"""99 — master schedule (ho_so/09_exp_plan.md): env (00, every launch) -> prep (01) -> per tier A, B, C: embeddings (02) then
model units (03, one process each) -> analysis (04) -> web export (05). Launch through run.bat.
Everything lands in Result/ (run folders, ThucNghiem_Log.csv, run_state.json, master_log.txt). A finished unit is recognised
from its run folder (metrics.csv present), run_state.json only caches speeds: the same command resumes after any
interruption, on any machine. After every unit one TIEN DO line (bar, units, hours left), also in the window title.
A failing unit stops the schedule with a CRASH entry + traceback in master_log.txt and exit code 1 (never swallowed).
  --tier A|B|C       run up to and including that tier (default C = everything)
  --budget-hours N   time-guard on the development laptop: no new unit after 80 % of N, then analysis; exit 75
  --estimate         print the time estimate and exit
  --smoke            end-to-end self-test (1 seed, 1 epoch, a subset of units) into Result_smoke/ (removed after, unless --keep)"""
import argparse, atexit, ctypes, os, shutil, subprocess, sys, time, traceback

import lib

S5, S3, S1 = [42, 43, 44, 45, 46], [42, 43, 44], [42]
# (dataset, model, embedding, seeds) — tiers exactly as ho_so/09_exp_plan.md
TIERS = {
    "A": [("emb", "beauty", "bge-base"), ("emb", "ml1m", "bge-base"),
          ("beauty", "popular", "-", S1), ("beauty", "random", "-", S1), ("beauty", "llmseqsim", "openai", S1), ("beauty", "llmseqsim", "bge-base", S1),
          ("beauty", "sasrec", "-", S5), ("beauty", "sasrec_l2s", "-", S5), ("beauty", "llm2sasrec", "palm", S5), ("beauty", "llm2sasrec", "openai", S5),
          ("beauty", "llm2sasrec", "bge-base", S5), ("beauty", "fused", "bge-base", S5),
          ("ml1m", "popular", "-", S1), ("ml1m", "random", "-", S1), ("ml1m", "llmseqsim", "bge-base", S1),
          ("ml1m", "sasrec", "-", S5), ("ml1m", "sasrec_l2s", "-", S5), ("ml1m", "llm2sasrec", "bge-base", S5),
          ("beauty", "llm2sasrec_shuf", "palm", S5)],   # new control class, last in tier A: a failure cannot block C1-C4
    "B": [("emb", "beauty", "minilm"), ("emb", "beauty", "bge-small"), ("emb", "ml1m", "minilm"), ("emb", "ml1m", "bge-small"),
          ("beauty", "llmseqsim", "palm", S1), ("beauty", "llmseqsim", "minilm", S1), ("beauty", "llmseqsim", "bge-small", S1),
          ("ml1m", "llmseqsim", "minilm", S1), ("ml1m", "llmseqsim", "bge-small", S1),
          ("beauty", "llm2sasrec", "minilm", S5), ("beauty", "fused", "openai", S5),
          ("beauty", "gru4rec", "-", S5), ("beauty", "vsknn", "-", S1),
          ("ml1m", "llm2sasrec", "minilm", S5), ("ml1m", "fused", "bge-base", S5), ("ml1m", "gru4rec", "-", S5), ("ml1m", "vsknn", "-", S1)],
    "C": [("ml1m", "bert4rec", "-", S3), ("ml1m", "llm2bert4rec", "bge-base", S3),
          ("beauty", "bert4rec", "-", S3), ("beauty", "llm2bert4rec", "palm", S3)],
}
SMOKE = [("emb", "ml1m", "bge-base"),
         ("ml1m", "popular", "-", S1), ("ml1m", "random", "-", S1), ("ml1m", "llmseqsim", "bge-base", S1), ("ml1m", "sasrec", "-", S1), ("ml1m", "sasrec_l2s", "-", S1),
         ("ml1m", "llm2sasrec", "bge-base", S1), ("ml1m", "fused", "bge-base", S1), ("ml1m", "gru4rec", "-", S1), ("ml1m", "vsknn", "-", S1),
         ("ml1m", "bert4rec", "-", S1), ("ml1m", "llm2bert4rec", "bge-base", S1),
         ("beauty", "sasrec", "-", S1), ("beauty", "llm2sasrec", "palm", S1), ("beauty", "llm2sasrec_shuf", "palm", S1), ("beauty", "llmseqsim", "openai", S1)]
# default seconds per unit, measured on the development laptop (i5-12450HX, CPU only) — ho_so/00c_kha_thi.md P2/P4;
# replaced by this machine's own measured speed as soon as a unit of the same kind finished here. BERT-family = s/epoch measured
# (Beauty 220/306, ML-1M LLM2BERT4Rec 165) x 8 epochs guessed; ML-1M V-SKNN 493 s measured (single process).
RATE = {"env": 30, "prep": 15, "ana": 120, "web": 60,
        "emb:beauty:bge-base": 420, "emb:beauty:minilm": 120, "emb:beauty:bge-small": 160, "emb:ml1m:bge-base": 80, "emb:ml1m:minilm": 30, "emb:ml1m:bge-small": 40,
        "beauty:random": 15, "beauty:llm2sasrec_shuf": 170, "ml1m:random": 10, "ml1m:llm2sasrec_shuf": 90,
        "beauty:popular": 30, "beauty:llmseqsim": 40, "beauty:sasrec": 100, "beauty:sasrec_l2s": 180, "beauty:llm2sasrec": 170, "beauty:fused": 190,
        "beauty:gru4rec": 130, "beauty:vsknn": 900, "beauty:bert4rec": 1900, "beauty:llm2bert4rec": 2600,
        "ml1m:popular": 15, "ml1m:llmseqsim": 20, "ml1m:sasrec": 45, "ml1m:sasrec_l2s": 90, "ml1m:llm2sasrec": 90, "ml1m:fused": 90,
        "ml1m:gru4rec": 60, "ml1m:vsknn": 500, "ml1m:bert4rec": 700, "ml1m:llm2bert4rec": 1400}
GUESSED = ("bert4rec", "llm2bert4rec")


class U:
    def __init__(self, key, kind, rate, cmd, tier):
        self.key, self.kind, self.rate, self.cmd, self.tier = key, kind, rate, cmd, tier

    def done(self, smoke):
        if self.kind == "emb":
            ds, emb = self.key.split(":")[1:]
            return (lib.ds_dir(smoke) / f"{ds}_emb_{emb}.csv.gz").exists() and lib.find_run(self.key, smoke) is not None
        if self.kind == "prep":                        # dataset/ is built once; rebuilding mid-campaign could change inputs
            d = lib.ds_dir(smoke)
            return all((d / f).exists() for f in ("beauty.pickle", "beauty_items.csv", "beauty_emb_openai.csv.gz", "beauty_emb_palm.csv.gz",
                                                  "ml1m.pickle", "ml1m_items.csv"))
        if self.kind == "rec":
            return lib.find_run(self.key, smoke) is not None
        return False                                   # env, ana, web: run on every launch (cheap)


def units(tier="C", smoke=False):
    """Ordered list: env, prep, then the tiers up to `tier` (embeddings of a tier before its models), then ana, web."""
    us = [U("env", "env", "env", ["00_kiem_tra_moi_truong.py"], "-"), U("prep", "prep", "prep", ["01_chuan_bi_du_lieu.py"], "-")]
    plan = [("S", SMOKE)] if smoke else [(t, TIERS[t]) for t in "ABC" if t <= tier]
    for t, items in plan:
        for it in items:
            if it[0] == "emb":
                us.append(U(f"emb:{it[1]}:{it[2]}", "emb", f"emb:{it[1]}:{it[2]}", ["02_tao_embedding.py", "--ds", it[1], "--emb", it[2]], t))
            else:
                ds, m, e, seeds = it
                for s in seeds:
                    us.append(U(f"rec:{ds}:{m}:{e}:{s}", "rec", f"{ds}:{m}", ["03_chay_mo_hinh.py", "--ds", ds, "--model", m, "--emb", e, "--seed", str(s)], t))
    us += [U("ana", "ana", "ana", ["04_phan_tich.py"], "-"), U("web", "web", "web", ["05_xuat_du_lieu_web.py"], "-")]
    return us


def speeds(st):
    """Own measured seconds per rate key (mean of finished units of that kind on this machine)."""
    acc = {}
    for d in st["done"].values():
        acc.setdefault(d["rate"], []).append(d["sec"])
    return {k: sum(v) / len(v) for k, v in acc.items()}


def left_hours(us, st, smoke, ran=()):
    own = speeds(st)
    return sum(own.get(u.rate, RATE[u.rate]) for u in us if u.key not in ran and not u.done(smoke)) / 3600


def other_run(procs, me, root):
    """pid of another 99_master_run.py working in `root` (procs: iterable of (pid, cmdline list, cwd))."""
    target = os.path.normcase(str(root / "99_master_run.py"))
    for pid, cmd, cwd in procs:
        if pid in me or not cwd or "--estimate" in cmd:
            continue
        if any(x.endswith("99_master_run.py") and os.path.normcase(os.path.join(cwd, x)) == target for x in cmd):
            return pid
    return None


def unattended_console():
    """Keep the PC awake while this process lives; switch QuickEdit off (a click would freeze the run) and restore it at exit."""
    if os.name != "nt":
        return
    k = ctypes.windll.kernel32
    k.SetThreadExecutionState(0x80000001)
    k.GetStdHandle.restype = ctypes.c_void_p
    h, mode = ctypes.c_void_p(k.GetStdHandle(-10)), ctypes.c_uint32()
    if k.GetConsoleMode(h, ctypes.byref(mode)):
        k.SetConsoleMode(h, (mode.value & ~0x40) | 0x80)
        atexit.register(k.SetConsoleMode, h, mode.value)


def title(text):
    if os.name == "nt":
        ctypes.windll.kernel32.SetConsoleTitleW(text)


def estimate(tier, smoke, st):
    us = units(tier, smoke)
    own = speeds(st)
    print(f"UOC LUONG thoi gian con lai | {lib.who()} | tang toi {tier}")
    for t in ("-", "A", "B", "C", "S"):
        sel = [u for u in us if u.tier == t]
        if sel:
            left = [u for u in sel if not u.done(smoke)]
            print(f"  tang {t:2s}: {len(sel) - len(left):3d}/{len(sel):3d} don vi xong | con ~{sum(own.get(u.rate, RATE[u.rate]) for u in left) / 3600:.1f} h")
    print(f"  TONG con ~{left_hours(us, st, smoke):.1f} h | " + (f"toc do da do tren may nay: {len(own)} loai don vi" if own else
          "toc do mac dinh = so do pilot tren laptop i5-12450HX (CPU); se tu hieu chinh"))
    print(f"  mac dinh CHUA DO (uoc): {', '.join(g for g in GUESSED if not any(k.endswith(g) for k in own))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="C", choices=["A", "B", "C"])
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep", action="store_true", help="smoke: keep Result_smoke/ and dataset_smoke/")
    ap.add_argument("--restart", action="store_true", help="forget run_state.json (finished units are still recognised from Result/)")
    ap.add_argument("--used-hours", type=float, default=0.0, help="hours already spent on this machine, counted against --budget-hours")
    a = ap.parse_args()
    smoke = a.smoke
    state_path = lib.res_dir(smoke) / "run_state.json"
    if a.estimate:
        return estimate(a.tier, smoke, lib.load_state(state_path))
    import psutil
    me = {os.getpid()} | {p.pid for p in psutil.Process().parents()}
    procs = []
    for p in psutil.process_iter(["pid", "cmdline", "cwd"]):
        procs.append((p.info["pid"], p.info["cmdline"] or [], p.info["cwd"]))
    if pid := other_run(procs, me, lib.ROOT):
        raise SystemExit(f"TU CHOI: dang co mot luot 99_master_run khac chay trong thu muc nay (pid {pid}).")
    if lib.ram_free_gb() < 2:
        raise SystemExit(f"TU CHOI: RAM trong {lib.ram_free_gb():.1f} GB < 2 GB — dong bot chuong trinh roi chay lai")
    if smoke:
        for p in (lib.res_dir(True), lib.ds_dir(True)):
            shutil.rmtree(p, ignore_errors=True)
    lib.res_dir(smoke).mkdir(exist_ok=True)
    st = {"done": {}} if a.restart else lib.load_state(state_path, tell=lambda m: lib.say(m, smoke))
    t0, limit = time.time() - a.used_hours * 3600, (0.8 * a.budget_hours * 3600 if a.budget_hours else None)
    lib.say(f"=== 99_master_run {' '.join(sys.argv[1:])} | {lib.who()} ===", smoke)
    unattended_console()
    us, ran, fired = units(a.tier, smoke), set(), False
    for i, u in enumerate(us):
        if u.done(smoke):
            continue
        if limit and u.kind in ("rec", "emb") and time.time() - t0 >= limit:
            if not fired:
                lib.say(f"!! time-guard: {(time.time() - t0) / 3600:.1f} h >= 80 % cua {a.budget_hours} h -> bo qua cac don vi con lai, chi phan tich", smoke)
            fired = True
            continue
        cmd = [sys.executable, str(lib.ROOT / u.cmd[0]), *u.cmd[1:]] + (["--smoke"] if smoke else [])
        lib.say(f">>> [{u.tier}] {u.key}", smoke)
        t = time.time()
        err = lib.res_dir(smoke) / "last_unit_stderr.txt"
        try:
            with open(err, "w", encoding="utf-8") as fe:          # stdout stays live; stderr kept so a crash shows the real error
                r = subprocess.run(cmd, cwd=lib.ROOT, stderr=fe)
            if r.returncode != 0:
                tail = "".join(err.read_text(encoding="utf-8", errors="replace").splitlines(True)[-30:])
                raise RuntimeError(f"{' '.join(u.cmd)} -> ma thoat {r.returncode}\n--- stderr (30 dong cuoi) ---\n{tail}")
            if u.kind in ("rec", "emb") and not u.done(smoke):
                raise RuntimeError(f"{u.key}: buoc thoat 0 nhung khong de lai san pham")
        except Exception as e:
            lib.say(f"CRASH don vi {u.key}: {e}\n{traceback.format_exc()}!! LICH DUNG. Sua nguyen nhan roi chay lai dung lenh cu — no chay tiep tu cho da dung.", smoke)
            title(f"SEQREC DA DUNG VI LOI o {u.key}")
            sys.exit(1)
        ran.add(u.key)
        st["done"][u.key] = {"rate": u.rate, "sec": round(time.time() - t, 1)}
        lib.save_state(state_path, st)
        n_done = sum(1 for x in us if x.key in ran or x.done(smoke))
        left = left_hours(us, st, smoke, ran)
        bar = "#" * (20 * n_done // len(us))
        lib.say(f"TIEN DO [{bar:<20s}] {n_done}/{len(us)} don vi | con ~{left:.1f} h | {lib.who()}", smoke)
        title(f"SEQREC {n_done}/{len(us)} don vi - con ~{left:.1f} h")
    lib.say(f"{'DUNG VI NGAN SACH (exit 75)' if fired else 'XONG'}: {(time.time() - t0) / 3600:.2f} h. Ket qua: {lib.res_dir(smoke)}", smoke)
    if smoke and not a.keep:
        for p in (lib.res_dir(True), lib.ds_dir(True)):
            shutil.rmtree(p, ignore_errors=True)
        print("smoke XONG: da xoa Result_smoke/ va dataset_smoke/ (--keep de giu lai)")
    if fired:
        sys.exit(75)


if __name__ == "__main__":
    main()
