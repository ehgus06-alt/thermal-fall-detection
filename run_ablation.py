"""
MHI-ablation runner (track 1).

For each input_mode x training seed, runs three evaluations and collects metrics,
then reports mean +- std across seeds per mode (JSON + Markdown table).

  modes : still, stack, no_mhi, full   (channel composition; see dataset.make_window_feat)
  seeds : training/init seeds (default 0,1,2). The train/val SPLIT is fixed (--split_seed 42)
          so every seed scores the SAME val set -> the +-std is pure run-to-run variance.

Per (mode, seed):
  1) in-domain : train.py --train_source all --eval_source all   -> clip bag max-pool AP/AUROC/F1/P/R
  2) streaming : eval_stream.py on that checkpoint                -> deployment AP/AUROC/Specificity/F1/R/P
  3) cross     : train.py --train_source flir --eval_source khan  -> cross-dataset clip-level AP/F1
  4) integ     : (opt-in, --integ) data-integration effect on the SAME test set:
                 BASE -> EVAL  vs  BASE+ADD -> EVAL   (default flir->khan vs flir+silver->khan)
                 BASE=flir/EVAL=khan reuses run 3's checkpoint. Default modes: full only.

Runs NOTHING by itself that needs no data - it shells out to train.py / eval_stream.py, so it
must run on the machine that has cache/ + torch. Smoke-test first: --epochs 2 --seeds 0.

Usage:
  python run_ablation.py                              # full: 4 modes x 3 seeds, 25 epochs
  python run_ablation.py --modes still full --seeds 0 1 2 --epochs 25
  python run_ablation.py --epochs 2 --seeds 0 --skip_existing   # quick smoke
  # add the integration experiment AFTER a main run (same modes/seeds; only the new runs train,
  # existing checkpoints and their streaming evals are reused):
  python run_ablation.py --modes still stack no_mhi full --seeds 0 1 2 --epochs 25 --skip_existing --integ
"""
import os, sys, json, argparse, subprocess, statistics
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "runs")
PY   = sys.executable


def run(cmd):
    print("\n$ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    return r.returncode == 0


def best_val_row(log_path):
    """Pick the best-AP epoch row from a train_log*.json (val runs only)."""
    with open(log_path, encoding="utf-8") as f:
        d = json.load(f)
    rows = [r for r in d.get("log", []) if "ap" in r]
    if not rows:
        return None
    return max(rows, key=lambda r: r["ap"])


def agg(values):
    vals = [v for v in values if v is not None and v == v]   # drop None / NaN
    if not vals:
        return dict(mean=None, std=None, n=0, raw=values)
    return dict(mean=round(statistics.mean(vals), 4),
                std=round(statistics.pstdev(vals) if len(vals) > 1 else 0.0, 4),
                n=len(vals), raw=[round(v, 4) for v in vals])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["still", "stack", "no_mhi", "full"],
                    choices=["still", "stack", "no_mhi", "full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--augment", dest="augment", action="store_true", default=True,
                    help="train-time augmentation (default ON, applied to every mode equally)")
    ap.add_argument("--no_augment", dest="augment", action="store_false")
    ap.add_argument("--cross", dest="cross", action="store_true", default=True,
                    help="also run flir->khan cross eval (default ON; needs both sources cached)")
    ap.add_argument("--no_cross", dest="cross", action="store_false")
    ap.add_argument("--skip_existing", action="store_true",
                    help="skip a training run that already FINISHED (checkpoint + train log), "
                         "and its streaming eval if that json is newer than the checkpoint; "
                         "a run killed mid-training (checkpoint but no log) is retrained")
    ap.add_argument("--integ", action="store_true",
                    help="data-integration experiment: BASE->EVAL vs BASE+ADD->EVAL on the same test set")
    ap.add_argument("--integ_base", default="flir", help="base training source (default flir)")
    ap.add_argument("--integ_add", default="silver", help="source added to the base (default silver)")
    ap.add_argument("--integ_eval", default="khan", help="held-out test source (default khan)")
    ap.add_argument("--integ_modes", nargs="+", default=["full"],
                    choices=["still", "stack", "no_mhi", "full"],
                    help="input modes to run the integration experiment for (default: full only)")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    runs = []   # one record per (mode, seed)
    for mode in args.modes:
        for seed in args.seeds:
            rec = dict(mode=mode, seed=seed)
            tag_in = f"_{mode}_s{seed}"
            ck_in  = os.path.join(OUT, f"best{tag_in}.pt")
            log_in = os.path.join(OUT, f"train_log{tag_in}.json")

            # 1) in-domain train + clip bag max-pool eval
            common = [PY, "train.py", "--input_mode", mode, "--seed", str(seed),
                      "--split_seed", str(args.split_seed), "--epochs", str(args.epochs)]
            if args.augment:
                common.append("--augment")
            # "done" = log exists: train.py saves best*.pt DURING training but writes the log only
            # at the end, so a killed run (ckpt but no log) is retrained instead of silently skipped
            if not (args.skip_existing and os.path.exists(ck_in) and os.path.exists(log_in)):
                if not run(common + ["--train_source", "all", "--eval_source", "all", "--tag", tag_in]):
                    print(f"!! in-domain train failed: {mode} s{seed}", flush=True)
            row = best_val_row(log_in) if os.path.exists(log_in) else None
            rec["indomain"] = ({k: row.get(k) for k in ("ap", "auroc", "f1", "precision", "recall")}
                               if row else None)

            # 2) streaming (deployment) eval on that checkpoint
            stream_json = os.path.join(OUT, f"stream{tag_in}.json")
            stream_fresh = (os.path.exists(stream_json) and os.path.exists(ck_in)
                            and os.path.getmtime(stream_json) >= os.path.getmtime(ck_in))
            if os.path.exists(ck_in) and not (args.skip_existing and stream_fresh):
                run([PY, "eval_stream.py", "--ckpt", ck_in,
                     "--split_seed", str(args.split_seed), "--json_out", stream_json])
            if os.path.exists(stream_json):
                with open(stream_json, encoding="utf-8") as f:
                    s = json.load(f)
                rf = s.get("recall_first", {})   # zero-miss/F2 deployment point (nested)
                rec["streaming"] = {k: s.get(k) for k in       # best-F1 operating point
                                    ("ap", "auroc", "specificity", "f1", "recall", "precision",
                                     "TP", "FP", "FN", "TN", "thr", "operating_point")}
                rec["streaming"].update(deploy_missed=rf.get("missed"), deploy_far=rf.get("far"),
                                        deploy_recall=rf.get("recall"), deploy_thr=rf.get("thr"))
            else:
                rec["streaming"] = None

            # 3) cross-dataset flir -> khan (clip bag max-pool AP)
            if args.cross:
                tag_x = f"_{mode}_s{seed}_cross"
                log_x = os.path.join(OUT, f"train_log{tag_x}.json")
                ck_x  = os.path.join(OUT, f"best{tag_x}.pt")
                if not (args.skip_existing and os.path.exists(ck_x) and os.path.exists(log_x)):
                    if not run(common + ["--train_source", "flir", "--eval_source", "khan", "--tag", tag_x]):
                        print(f"!! cross train failed (need flir+khan cached): {mode} s{seed}", flush=True)
                rowx = best_val_row(log_x) if os.path.exists(log_x) else None
                rec["cross_flir2khan"] = ({k: rowx.get(k) for k in ("ap", "auroc", "f1", "precision", "recall")}
                                          if rowx else None)

            # 4) data-integration: BASE -> EVAL  vs  BASE+ADD -> EVAL (same held-out test source)
            if args.integ and mode in args.integ_modes:
                ev = args.integ_eval
                rec["integ"] = {}
                for key, src in (("base", args.integ_base),
                                 ("integrated", f"{args.integ_base}+{args.integ_add}")):
                    toks = sorted(t for t in src.split("+") if t)
                    if toks == ["flir"] and ev == "khan":
                        tag = f"_{mode}_s{seed}_cross"                          # identical to run 3 -> reuse
                    else:
                        tag = f"_{mode}_s{seed}_x_{'-'.join(toks)}_to_{ev}"     # order-independent name
                    ck = os.path.join(OUT, f"best{tag}.pt")
                    lg = os.path.join(OUT, f"train_log{tag}.json")
                    done_above = tag.endswith("_cross") and args.cross
                    if not done_above and not (args.skip_existing and os.path.exists(ck) and os.path.exists(lg)):
                        if not run(common + ["--train_source", "+".join(toks), "--eval_source", ev, "--tag", tag]):
                            print(f"!! integration train failed: {src}->{ev} {mode} s{seed}", flush=True)
                    r = best_val_row(lg) if os.path.exists(lg) else None
                    rec["integ"][key] = dict(train_source="+".join(toks), eval_source=ev,
                                             **({k: r.get(k) for k in ("ap", "auroc", "f1", "precision", "recall")}
                                                if r else {}))
            runs.append(rec)

    # ---- aggregate mean +- std per mode ----
    summary = {}
    for mode in args.modes:
        rs = [r for r in runs if r["mode"] == mode]
        def col(section, key):
            return agg([(r[section] or {}).get(key) if r.get(section) else None for r in rs])
        summary[mode] = dict(
            indomain=dict(ap=col("indomain", "ap"), auroc=col("indomain", "auroc"),
                          f1=col("indomain", "f1"), precision=col("indomain", "precision"),
                          recall=col("indomain", "recall")),
            streaming=dict(ap=col("streaming", "ap"), auroc=col("streaming", "auroc"),
                           specificity=col("streaming", "specificity"), f1=col("streaming", "f1"),
                           recall=col("streaming", "recall"), precision=col("streaming", "precision")),
            cross=dict(ap=col("cross_flir2khan", "ap"), f1=col("cross_flir2khan", "f1")),
        )

    # ---- data-integration summary (paired by seed: delta = integrated - base) ----
    integration = None
    if args.integ:
        integration = {}
        for mode in args.integ_modes:
            rs = [r for r in runs if r["mode"] == mode and r.get("integ")]
            if not rs:
                continue
            def icol(which, key):
                return agg([(r["integ"].get(which) or {}).get(key) for r in rs])
            deltas = []
            for r in rs:
                b = (r["integ"].get("base") or {}).get("ap")
                i = (r["integ"].get("integrated") or {}).get("ap")
                deltas.append(i - b if (b is not None and i is not None) else None)
            integration[mode] = dict(
                eval_source=args.integ_eval,
                base=dict(train_source=rs[0]["integ"]["base"]["train_source"],
                          ap=icol("base", "ap"), auroc=icol("base", "auroc"), f1=icol("base", "f1")),
                integrated=dict(train_source=rs[0]["integ"]["integrated"]["train_source"],
                                ap=icol("integrated", "ap"), auroc=icol("integrated", "auroc"),
                                f1=icol("integrated", "f1")),
                delta_ap=agg(deltas))

    result = dict(created=datetime.now().isoformat(timespec="seconds"),
                  config=vars(args), runs=runs, summary=summary, integration=integration)
    out_json = os.path.join(OUT, "ablation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # ---- Markdown table (paper table 3) ----
    def cell(a):
        return "-" if a["mean"] is None else f"{a['mean']:.3f}±{a['std']:.3f}"
    lines = []
    lines.append("| mode | in-AP | in-AUROC | in-F1 | stream-AP | stream-AUROC | stream-Spec | stream-R | cross-AP |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for mode in args.modes:
        s = summary[mode]
        lines.append("| {m} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} |".format(
            m=mode,
            a=cell(s["indomain"]["ap"]), b=cell(s["indomain"]["auroc"]), c=cell(s["indomain"]["f1"]),
            d=cell(s["streaming"]["ap"]), e=cell(s["streaming"]["auroc"]),
            f=cell(s["streaming"]["specificity"]), g=cell(s["streaming"]["recall"]),
            h=cell(s["cross"]["ap"])))
    if integration:
        lines += ["", f"## Data integration (test = {args.integ_eval})", "",
                  "| mode | train | AP | AUROC | F1 |", "|---|---|---|---|---|"]
        for mode, s in integration.items():
            for key in ("base", "integrated"):
                lines.append(f"| {mode} | {s[key]['train_source']} | {cell(s[key]['ap'])} | "
                             f"{cell(s[key]['auroc'])} | {cell(s[key]['f1'])} |")
            lines.append(f"| {mode} | delta AP (integrated - base, paired by seed) | {cell(s['delta_ap'])} | | |")
    table = "\n".join(lines)
    out_md = os.path.join(OUT, "ablation_table.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# MHI ablation (seeds={args.seeds}, split_seed={args.split_seed}, "
                f"epochs={args.epochs}, augment={args.augment})\n\n{table}\n")

    print("\n" + table)
    print(f"\n[done] {out_json}\n       {out_md}", flush=True)


if __name__ == "__main__":
    main()
