from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

from rover_sim.config import CASE_LABELS, SCENARIO_LABELS, SimConfig
from rover_sim.simulation import run_simulation


def _numeric_keys(rows):
    skip = {"case", "scenario", "seed", "path"}
    return [k for k, v in rows[0].items() if k not in skip and isinstance(v, (int, float))]


def aggregate(rows):
    out = []
    for scenario in sorted({r["scenario"] for r in rows}):
        for case in sorted({r["case"] for r in rows}):
            rr = [r for r in rows if r["scenario"] == scenario and r["case"] == case]
            if not rr: continue
            row = {"scenario": scenario, "case": case, "n": len(rr)}
            for k in _numeric_keys(rr):
                vals = [float(r[k]) for r in rr if np.isfinite(float(r[k]))]
                if vals:
                    row[k+"_mean"] = mean(vals)
                    row[k+"_std"] = pstdev(vals) if len(vals) > 1 else 0.0
            out.append(row)
    return out


def write_csv(path: Path, rows):
    if not rows: return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader(); wr.writerows(rows)


def make_summary_plots(out: Path, agg, scenarios, cases):
    import matplotlib.pyplot as plt
    specs = [
        ("rmse_cross_track_m_mean", "Cross-track RMSE [m]", "comparison_tracking_rmse.png"),
        ("rms_slip_mean", "Wheel-slip RMS [-]", "comparison_slip_rms.png"),
        ("specific_energy_J_per_m_mean", "Specific energy proxy [J/m]", "comparison_specific_energy.png"),
        ("max_normal_load_N_mean", "Peak normal load [N]", "comparison_peak_load.png"),
    ]
    x = np.arange(len(scenarios)); width = 0.18
    for metric, ylabel, filename in specs:
        fig = plt.figure(figsize=(9.0, 4.8))
        for j, case in enumerate(cases):
            vals, errs = [], []
            for s in scenarios:
                r = next((q for q in agg if q["scenario"] == s and q["case"] == case), None)
                vals.append(np.nan if r is None else r.get(metric, np.nan))
                errs.append(0.0 if r is None else r.get(metric.replace("_mean","_std"), 0.0))
            plt.bar(x + (j-(len(cases)-1)/2)*width, vals, width, yerr=errs, capsize=2, label=case)
        plt.xticks(x, scenarios); plt.ylabel(ylabel); plt.grid(True, axis="y", alpha=0.25); plt.legend(ncol=4)
        fig.tight_layout(); fig.savefig(out / filename, dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Batch C0-C3 ablation and robustness study")
    ap.add_argument("--cases", nargs="+", choices=list(CASE_LABELS), default=["C0","C1","C2","C3"])
    ap.add_argument("--scenarios", nargs="+", choices=list(SCENARIO_LABELS), default=["nominal","rough","low_mu","split_mu","disturbance"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 11, 19])
    ap.add_argument("--path", default="s_curve")
    ap.add_argument("--duration", type=float, default=0.0, help="0 = automatic time budget")
    ap.add_argument("--speed", type=float, default=0.30)
    ap.add_argument("--output", type=Path, default=Path("results/journal_suite"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in args.scenarios:
        for case in args.cases:
            for seed in args.seeds:
                cfg = SimConfig()
                cfg.gui = False; cfg.realtime = False; cfg.arm_enabled = False
                cfg.case = case; cfg.scenario = scenario; cfg.path_name = args.path; cfg.seed = seed; cfg.terrain.seed = seed
                cfg.duration = args.duration; cfg.control.desired_speed = args.speed
                cfg.output_dir = args.output / "runs" / f"{case}_{scenario}_seed{seed}"
                print(f"\n>>> {case} / {scenario} / seed={seed}")
                metrics = run_simulation(cfg, verbose=False)
                row = {"case": case, "scenario": scenario, "seed": seed, "path": args.path, **metrics}
                rows.append(row)
                print({k: round(v,4) if isinstance(v,float) and np.isfinite(v) else v for k,v in metrics.items()})
                write_csv(args.output / "raw_metrics.csv", rows)

    agg = aggregate(rows)
    write_csv(args.output / "aggregate_metrics.csv", agg)
    make_summary_plots(args.output, agg, args.scenarios, args.cases)
    with (args.output / "suite_config.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"\nJournal suite saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
