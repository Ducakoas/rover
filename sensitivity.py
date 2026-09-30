"""Optional sensitivity studies for C3 journal revision responses.

Study A: QP slip-growth penalty on split-mu terrain.
Study B: common FTDO gain scale on injected-disturbance terrain.
"""
from pathlib import Path
import csv
from rover_sim.config import SimConfig
from rover_sim.simulation import run_simulation

OUT = Path("results/sensitivity")
OUT.mkdir(parents=True, exist_ok=True)
rows = []

for growth in [10.0, 28.0, 50.0]:
    cfg = SimConfig(); cfg.gui=False; cfg.realtime=False; cfg.arm_enabled=False
    cfg.case="C3"; cfg.scenario="split_mu"; cfg.control.qp_slip_growth=growth
    cfg.output_dir = OUT / f"qp_slip_growth_{growth:g}"
    m = run_simulation(cfg, verbose=False)
    rows.append({"study":"qp_slip_growth","value":growth,**m})

for scale in [0.75, 1.0, 1.25]:
    cfg = SimConfig(); cfg.gui=False; cfg.realtime=False; cfg.arm_enabled=False
    cfg.case="C3"; cfg.scenario="disturbance"
    cfg.control.ftdo_a1 *= scale; cfg.control.ftdo_a2 *= scale
    cfg.control.ftdo_b1 *= scale; cfg.control.ftdo_b2 *= scale
    cfg.output_dir = OUT / f"ftdo_gain_scale_{scale:g}"
    m = run_simulation(cfg, verbose=False)
    rows.append({"study":"ftdo_gain_scale","value":scale,**m})

keys=[]
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
with (OUT/"sensitivity_metrics.csv").open("w",newline="",encoding="utf-8") as f:
    wr=csv.DictWriter(f,fieldnames=keys); wr.writeheader(); wr.writerows(rows)
print("Saved", (OUT/"sensitivity_metrics.csv").resolve())
