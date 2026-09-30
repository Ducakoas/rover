# Methods and equations matching the code

This file is intended to keep manuscript equations synchronized with the Python implementation.

## 1. Path tracking

For look-ahead distance Ld and heading-to-target error alpha, the Pure Pursuit curvature is

kappa = 2 sin(alpha) / Ld.

The instantaneous center radius is R = 1/kappa. With half wheelbase L and half track d, four-corner steering is approximated as

delta_LF = atan2(L, R-d),
delta_RF = atan2(L, R+d),
delta_LR = -delta_LF,
delta_RR = -delta_RF.

For wheel-center position (x_i,y_i), the ICR radius is

r_i = sqrt(x_i^2 + (R-y_i)^2),

and the nominal wheel angular speed is

omega_i* = v* r_i / (|R| R_w).

For near-straight driving omega_i* = v*/R_w.

## 2. Wheel slip

The code uses a symmetric low-speed-safe slip definition

lambda_i = (R_w omega_i - v_{x,i}) / max(|v_{x,i}|, |R_w omega_i|, v_eps).

This differs from some tire conventions; state the exact definition in the paper.

## 3. Longitudinal nominal model used by the observer

vdot_x = f_nom + d,

f_nom = eta_d sum(T_i)/(m R_w) - g sin(theta_s) - a_rr sign(T_sum).

Here theta_s is estimated from the local procedural terrain gradient along the rover heading. All unmodeled contact, drag and rigid-terrain effects are lumped into d.

## 4. Fixed-time disturbance observer

Let e_o = z_0 - v_x and sig^p(e)=|e|^p sign(e). The implemented observer is

z0_dot = f_nom + z1 - a1 sig^alpha(e_o) - a2 sig^beta(e_o),

z1_dot = -b1 sig^(2 alpha - 1)(e_o) - b2 sig^(2 beta - 1)(e_o),

d_hat = z1.

Default gains:
alpha=0.75, beta=1.50, a1=3.8, a2=0.08, b1=6.0, b2=0.10.

The implementation clips z1 and uses a warm-up interval. Do not claim exact fixed-time convergence of the entire PyBullet closed loop. If a theorem is included, scope it to the reduced observer error model and state its assumptions explicitly.

## 5. Body-speed command

e_v = v* - v_x,

a_des = sat(Kp e_v + Ki integral(e_v), +/- a_max).

The total wheel torque demand is

T_req = m R_w (a_des - d_hat) / eta_d.

For C1, d_hat=0. For C2/C3, d_hat comes from the FTDO.

## 6. C1/C2 heuristic allocation

Define slip excess

q_i = max(|lambda_i|-lambda_th,0),

g_i = 1/(1+k_lambda q_i),

w_i = g_i / sum_j g_j.

Then

T_i = T_req w_i + k_sync(omega_i* - omega_i),

followed by torque saturation.

## 7. C3 convex QP allocation

Decision vector T=[T_1,...,T_6]^T.

The implemented objective is

J(T) = w_F(1^T T - T_req)^2
     + sum_i w_{lambda,i} T_i^2
     + w_sync ||T-T_sync||_2^2
     + w_delta ||T-T_prev||_2^2,

where

w_{lambda,i} = w_lambda [1 + k_q max(|lambda_i|-lambda_th,0)^2],

T_sync = (T_req/6)1 + k_sync(omega* - omega).

Traction-aware box constraints are

|T_i| <= min(T_max, eta_mu mu_i Fz_i R_w).

If contact force is nearly zero, the torque cap is additionally reduced to the airborne-wheel limit.

The problem is a convex box-constrained quadratic program. It is solved with deterministic projected-gradient iterations using a step below 1/lambda_max(H).

## 8. Metrics

Main recommended metrics:
- RMSE and max cross-track error
- heading RMSE
- speed RMSE
- RMS and peak wheel slip
- cumulative absolute slip
- RMS and peak torque
- max torque rate
- peak wheel normal load and normal-load coefficient of variation
- contact-loss fraction
- roll/pitch envelope
- FTDO RMSE versus injected disturbance in the disturbance scenario
- integral of |T_i omega_i| as an electrical/mechanical energy proxy
- energy proxy divided by traveled distance

## 9. Modeling limitation to state explicitly

PyBullet provides rigid-body contact and Coulomb-type friction. The simulation does not model deformable Martian regolith, sinkage, bulldozing resistance or pressure-sinkage terramechanics. Therefore use language such as "rigid heterogeneous uneven terrain" rather than "high-fidelity Martian soil".
