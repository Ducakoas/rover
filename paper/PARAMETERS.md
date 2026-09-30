# Default parameters in the supplied implementation

## Rover
- wheel radius: 0.215 m
- wheel width: 0.255 m
- half track: 0.70 m
- half wheelbase: 0.95 m
- nominal total mass: 180 kg
- max wheel torque: 35 N m
- max steering angle: 38 deg
- Mars gravity: 3.71 m/s^2

## Control
- desired speed: 0.30 m/s
- lookahead: 1.35 m
- slip threshold: 0.12
- body-speed PI: Kp=1.55, Ki=0.38

## FTDO
- alpha=0.75
- beta=1.50
- a1=3.8
- a2=0.08
- b1=6.0
- b2=0.10
- estimate clip: +/-2.0 m/s^2
- warmup: 0.8 s

## QP
- force tracking weight: 8.0
- wheel-speed sync weight: 0.35
- torque smoothness weight: 0.60
- base slip penalty: 0.08
- slip penalty growth: 28.0
- projected-gradient iterations: 28 per control update
- traction safety factor: 0.90

These are starting values, not experimentally validated rover parameters. Report them exactly if used, and perform sensitivity/robustness analysis before submission.
