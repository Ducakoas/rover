# Journal experiment plan

## Core hypotheses

H1. Slip-aware allocation (C1) reduces RMS/peak wheel slip relative to C0 on low-friction and heterogeneous-friction terrain.

H2. FTDO assistance (C2) reduces speed error and disturbance-rejection error relative to C1 under the injected disturbance scenario.

H3. Convex traction-constrained QP allocation (C3) improves the trade-off among wheel slip, peak normal-load/traction demand, torque smoothness and energy proxy relative to C2 without materially degrading path tracking.

Do not predetermine which case must win every metric.

## Cases

C0: Pure Pursuit + Ackermann + 6 wheel PI.

C1: same path loop + slip-aware heuristic torque allocation.

C2: C1 + FTDO.

C3: FTDO + convex six-wheel QP allocation.

## Primary scenarios

1. nominal
2. rough
3. low_mu
4. split_mu
5. disturbance

Optional robustness scenarios: slope, patchy_mu, rock_field.

## Repetition

Minimum exploratory study: seeds = [7,11,19].

For a stronger journal submission: use >=10 terrain seeds for stochastic scenarios and report mean +/- standard deviation or confidence intervals. Determine the sample count before looking at final significance results.

## Statistical analysis

Recommended pairwise comparison within each scenario because every case uses the same seed/terrain realization. Possible choices:
- paired t-test if metric differences are approximately normal;
- Wilcoxon signed-rank test otherwise;
- report effect size and confidence intervals, not p-values alone.

The supplied code exports raw per-run metrics; statistical testing is intentionally not hard-coded so the final method can match the target journal's requirements.

## Required plots

Per representative run:
1. reference vs actual trajectory
2. cross-track error
3. longitudinal speed
4. all six wheel slips
5. all six wheel torques
6. roll/pitch
7. injected disturbance vs FTDO estimate
8. normal wheel loads
9. friction coefficients
10. energy proxy

Across runs:
- C0-C3 RMSE cross-track
- C0-C3 RMS slip
- C0-C3 specific energy proxy
- C0-C3 peak wheel load
- disturbance observer RMSE for C2/C3

## Failure criteria

Define before the final experiment campaign. Suggested examples:
- roll or pitch magnitude > 45 deg;
- rover body height below a geometry-based threshold;
- final goal distance above 2 m after full horizon;
- numerical disconnection/crash.

Do not silently remove failed runs. Count and report failure rate.
