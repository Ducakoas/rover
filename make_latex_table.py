from pathlib import Path
import argparse
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument("--csv", type=Path, default=Path("results/journal_suite/aggregate_metrics.csv"))
ap.add_argument("--out", type=Path, default=Path("results/journal_suite/table_main.tex"))
args=ap.parse_args()
df=pd.read_csv(args.csv)
cols=["scenario","case","rmse_cross_track_m_mean","rms_slip_mean","specific_energy_J_per_m_mean","max_normal_load_N_mean"]
missing=[c for c in cols if c not in df.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}\nRun compare.py first.")
t=df[cols].copy()
t.columns=["Scenario","Case","CTE RMSE [m]","Slip RMS [-]","Specific energy [J/m]","Peak load [N]"]
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(t.to_latex(index=False,float_format=lambda x:f"{x:.3f}",escape=False),encoding="utf-8")
print("Saved",args.out.resolve())
