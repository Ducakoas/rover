# V3 journal-oriented changes

Compared with the previous V2 demo:

- replaced simple sinusoidal terrain with deterministic multi-scale Mars-like procedural heightfields, crater/ridge structure and optional rocks;
- added homogeneous, rough, slope, low-mu, split-mu, patchy-mu, rock-field and disturbance scenarios;
- retained NASA-origin rover visual meshes and simplified robust collision geometry;
- kept Pure Pursuit + four-corner Ackermann path tracking as the common outer loop;
- defined a formal C0/C1/C2/C3 ablation structure;
- implemented a two-state fixed-time disturbance observer;
- implemented convex six-wheel traction-constrained QP allocation using a deterministic projected-gradient solver;
- added external disturbance ground truth for observer evaluation;
- improved wheel slip definition around zero speed;
- added wheel normal load, contact flag, friction, suspension state and torque-cap logging;
- expanded metrics and generated ten paper-ready per-run plots;
- added multi-seed journal comparison suite and aggregate CSV/plots;
- added parameter sensitivity script;
- added LaTeX result-table exporter;
- added manuscript template, equations, experiment plan, figure/table map, limitations and BibTeX references;
- added static project validation and unit tests;
- deliberately excluded MathWorks-specific terrain/path files because of their license terms.
