from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np

from .config import RoverConfig, STEER_ORDER, WHEEL_ORDER


class RoverPlant:
    def __init__(self, pb, urdf_path: Path, start_pos, start_yaw: float, cfg: RoverConfig):
        self.pb = pb
        self.cfg = cfg

        quat = pb.getQuaternionFromEuler([0.0, 0.0, float(start_yaw)])
        if hasattr(pb, "setAdditionalSearchPath"):
            pb.setAdditionalSearchPath(str(urdf_path.parent))

        self.body = pb.loadURDF(
            str(urdf_path),
            start_pos,
            quat,
            useFixedBase=False,
        )

        self.joint_by_name: Dict[str, int] = {}
        self.link_by_name: Dict[str, int] = {}

        for j in range(pb.getNumJoints(self.body)):
            info = pb.getJointInfo(self.body, j)
            self.joint_by_name[info[1].decode("utf-8")] = j
            self.link_by_name[info[12].decode("utf-8")] = j

        self.wheel_joints = {
            w: self.joint_by_name[f"{w.lower()}_wheel_joint"]
            for w in WHEEL_ORDER
        }
        self.wheel_links = {
            w: self.link_by_name[f"{w.lower()}_wheel"]
            for w in WHEEL_ORDER
        }
        self.steer_joints = {
            w: self.joint_by_name[f"{w.lower()}_steer_joint"]
            for w in STEER_ORDER
        }

        self.rocker_joints = [
            self.joint_by_name["left_rocker_joint"],
            self.joint_by_name["right_rocker_joint"],
        ]
        self.bogie_joints = [
            self.joint_by_name["left_bogie_joint"],
            self.joint_by_name["right_bogie_joint"],
        ]

        self.arm_joints = [
            self.joint_by_name["arm_yaw_joint"],
            self.joint_by_name["arm_shoulder_joint"],
            self.joint_by_name["arm_elbow_joint"],
            self.joint_by_name["arm_roll_joint"],
            self.joint_by_name["arm_wrist_joint"],
            self.joint_by_name["arm_wrist_yaw_joint"],
        ]

        self.gripper_joints = [
            self.joint_by_name[name]
            for name in ("gripper_left_joint", "gripper_right_joint")
            if name in self.joint_by_name
        ]

        self.ee_link = self.link_by_name["arm_ee"]

        self._wheel_hold_active = False
        self._disable_default_motors()
        self._configure_links()

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _disable_default_motors(self) -> None:
        p = self.pb
        free_joints = (
            list(self.wheel_joints.values())
            + self.rocker_joints
            + self.bogie_joints
            + self.arm_joints
            + self.gripper_joints
        )
        for idx in free_joints:
            p.setJointMotorControl2(
                self.body,
                idx,
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
            )

    def _configure_links(self) -> None:
        p = self.pb
        p.changeDynamics(
            self.body,
            -1,
            linearDamping=0.02,
            angularDamping=0.04,
        )

        for idx in self.wheel_links.values():
            p.changeDynamics(
                self.body,
                idx,
                lateralFriction=0.9,
                spinningFriction=0.02,
                rollingFriction=0.010,
                restitution=0.0,
                linearDamping=0.01,
                angularDamping=0.01,
            )

    def actual_total_mass(self) -> float:
        total = float(self.pb.getDynamicsInfo(self.body, -1)[0])
        for j in range(self.pb.getNumJoints(self.body)):
            total += float(self.pb.getDynamicsInfo(self.body, j)[0])
        return total

    # ------------------------------------------------------------------
    # Rover state
    # ------------------------------------------------------------------

    def pose(self):
        pos, q = self.pb.getBasePositionAndOrientation(self.body)
        rpy = self.pb.getEulerFromQuaternion(q)
        return (
            np.asarray(pos, float),
            np.asarray(rpy, float),
            np.asarray(q, float),
        )

    def body_velocity(self):
        lin, ang = self.pb.getBaseVelocity(self.body)
        _, _, q = self.pose()
        R = np.asarray(
            self.pb.getMatrixFromQuaternion(q),
            dtype=float,
        ).reshape(3, 3)
        return (
            R.T @ np.asarray(lin, dtype=float),
            R.T @ np.asarray(ang, dtype=float),
        )

    def wheel_omegas(self) -> np.ndarray:
        return np.array(
            [
                self.pb.getJointState(self.body, self.wheel_joints[w])[1]
                for w in WHEEL_ORDER
            ],
            dtype=float,
        )

    def steer_angles(self) -> np.ndarray:
        return np.array(
            [
                self.pb.getJointState(self.body, self.steer_joints[w])[0]
                for w in STEER_ORDER
            ],
            dtype=float,
        )

    def suspension_angles(self) -> tuple[float, float, float, float]:
        q = [
            float(self.pb.getJointState(self.body, j)[0])
            for j in self.rocker_joints + self.bogie_joints
        ]
        return q[0], q[1], q[2], q[3]

    def arm_joint_positions(self) -> np.ndarray:
        return np.asarray(
            [float(self.pb.getJointState(self.body, j)[0]) for j in self.arm_joints],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Wheel/contact state
    # ------------------------------------------------------------------

    def wheel_world_states(self):
        out = {}
        for w in WHEEL_ORDER:
            st = self.pb.getLinkState(
                self.body,
                self.wheel_links[w],
                computeLinkVelocity=1,
            )
            pos = np.asarray(st[0], float)
            quat = st[1]
            lin = np.asarray(st[6], float)
            R = np.asarray(
                self.pb.getMatrixFromQuaternion(quat),
                dtype=float,
            ).reshape(3, 3)

            axle = R[:, 1]
            forward = np.cross(
                axle,
                np.array([0.0, 0.0, 1.0]),
            )
            n = float(np.linalg.norm(forward))
            if n < 1e-8:
                forward = np.array([1.0, 0.0, 0.0])
            else:
                forward = forward / n

            out[w] = (pos, lin, forward)

        return out

    def wheel_longitudinal_speeds(self) -> np.ndarray:
        states = self.wheel_world_states()
        return np.asarray(
            [float(states[w][1] @ states[w][2]) for w in WHEEL_ORDER],
            dtype=float,
        )

    def slips(self) -> np.ndarray:
        vlong = self.wheel_longitudinal_speeds()
        omega = self.wheel_omegas()
        circum = self.cfg.wheel_radius * omega

        den = np.maximum.reduce(
            [
                np.abs(vlong),
                np.abs(circum),
                np.ones(6) * 0.06,
            ]
        )
        slip = (circum - vlong) / den
        return np.clip(slip, -1.5, 1.5)

    def set_spatial_friction(self, terrain) -> np.ndarray:
        mus = []
        states = self.wheel_world_states()

        for w in WHEEL_ORDER:
            pos = states[w][0]
            mu = float(
                terrain.friction_at(
                    float(pos[0]),
                    float(pos[1]),
                )
            )
            mus.append(mu)
            self.pb.changeDynamics(
                self.body,
                self.wheel_links[w],
                lateralFriction=mu,
            )

        return np.asarray(mus, dtype=float)

    def normal_forces(self, terrain_id: int) -> np.ndarray:
        fz = []
        for w in WHEEL_ORDER:
            cps = self.pb.getContactPoints(
                bodyA=self.body,
                bodyB=terrain_id,
                linkIndexA=self.wheel_links[w],
            )
            fz.append(float(sum(cp[9] for cp in cps)))
        return np.asarray(fz, dtype=float)

    def contact_flags(self, terrain_id: int) -> np.ndarray:
        return np.asarray(
            [
                1.0
                if self.pb.getContactPoints(
                    bodyA=self.body,
                    bodyB=terrain_id,
                    linkIndexA=self.wheel_links[w],
                )
                else 0.0
                for w in WHEEL_ORDER
            ],
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Steering / traction / suspension commands
    # ------------------------------------------------------------------

    def command_steering(self, steer: Dict[str, float]) -> None:
        p = self.pb
        for w, target in steer.items():
            p.setJointMotorControl2(
                self.body,
                self.steer_joints[w],
                p.POSITION_CONTROL,
                targetPosition=float(target),
                force=self.cfg.max_steer_force,
                positionGain=self.cfg.steer_position_gain,
                velocityGain=self.cfg.steer_velocity_gain,
                maxVelocity=3.0,
            )

    def release_wheels(self) -> None:
        """Disable the zero-speed wheel hold before returning to torque drive.

        PyBullet's velocity motor used by :meth:`hold_wheels` is intentionally
        strong.  A random-sample mission waits several seconds with that motor
        enabled.  Explicitly disabling it before TORQUE_CONTROL prevents the
        hold motor from continuing to pin the six wheels at zero speed after
        online replanning.
        """
        if not self._wheel_hold_active:
            return
        p = self.pb
        for w in WHEEL_ORDER:
            p.setJointMotorControl2(
                self.body,
                self.wheel_joints[w],
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=0.0,
            )
        self._wheel_hold_active = False

    def command_wheel_torques(self, torques: np.ndarray) -> None:
        # Random-sample missions deliberately hold the wheels while the sample
        # falls.  Always release that motor before applying drive torque.
        self.release_wheels()
        p = self.pb
        for w, tau in zip(WHEEL_ORDER, np.asarray(torques, dtype=float)):
            p.setJointMotorControl2(
                self.body,
                self.wheel_joints[w],
                p.TORQUE_CONTROL,
                force=float(tau),
            )

    def hold_wheels(self, force: float | None = None) -> None:
        """Actively hold all wheels at zero speed during waiting/arm phases."""
        p = self.pb
        hold_force = float(self.cfg.wheel_hold_force if force is None else force)
        for w in WHEEL_ORDER:
            p.setJointMotorControl2(
                self.body,
                self.wheel_joints[w],
                p.VELOCITY_CONTROL,
                targetVelocity=0.0,
                force=hold_force,
                velocityGain=1.0,
            )
        self._wheel_hold_active = True

    def command_passive_differential(self) -> None:
        p = self.pb
        ql, qdl = p.getJointState(self.body, self.rocker_joints[0])[:2]
        qr, qdr = p.getJointState(self.body, self.rocker_joints[1])[:2]

        # Differential coupling: mirrored rockers should have approximately
        # opposite joint angles, hence q_left + q_right -> 0.
        tau = (
            -self.cfg.rocker_diff_k * (ql + qr)
            -self.cfg.rocker_diff_c * (qdl + qdr)
        )
        tau = float(np.clip(tau, -70.0, 70.0))

        p.setJointMotorControl2(
            self.body,
            self.rocker_joints[0],
            p.TORQUE_CONTROL,
            force=tau,
        )
        p.setJointMotorControl2(
            self.body,
            self.rocker_joints[1],
            p.TORQUE_CONTROL,
            force=tau,
        )

    # ------------------------------------------------------------------
    # Manipulator
    # ------------------------------------------------------------------

    def command_arm_joint_targets(
        self,
        targets: Iterable[float],
        force: float = 75.0,
        max_velocity: float = 1.25,
    ) -> None:
        p = self.pb
        for idx, qd in zip(self.arm_joints, targets):
            p.setJointMotorControl2(
                self.body,
                idx,
                p.POSITION_CONTROL,
                targetPosition=float(qd),
                force=float(force),
                positionGain=0.32,
                velocityGain=0.12,
                maxVelocity=float(max_velocity),
            )

    def command_arm_ik(self, target_position, target_orientation=None) -> np.ndarray:
        p = self.pb

        kwargs = dict(
            targetPosition=list(map(float, target_position)),
            maxNumIterations=120,
            residualThreshold=5e-5,
        )
        if target_orientation is not None:
            kwargs["targetOrientation"] = list(map(float, target_orientation))

        q_all = p.calculateInverseKinematics(
            self.body,
            self.ee_link,
            **kwargs,
        )

        movable = []
        for j in range(p.getNumJoints(self.body)):
            jt = p.getJointInfo(self.body, j)[2]
            if jt in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                movable.append(j)

        q_map = {j: q for j, q in zip(movable, q_all)}
        targets = [
            q_map.get(j, p.getJointState(self.body, j)[0])
            for j in self.arm_joints
        ]

        self.command_arm_joint_targets(targets)
        return np.asarray(targets, dtype=float)

    def command_gripper(self, open_fraction: float) -> None:
        """Open/close the V9 two-finger visual/physical gripper.

        open_fraction = 0 -> closed
        open_fraction = 1 -> open
        """
        if not self.gripper_joints:
            return

        p = self.pb
        q = float(np.clip(open_fraction, 0.0, 1.0)) * 0.045
        for idx in self.gripper_joints:
            p.setJointMotorControl2(
                self.body,
                idx,
                p.POSITION_CONTROL,
                targetPosition=q,
                force=24.0,
                positionGain=0.45,
                velocityGain=0.15,
                maxVelocity=0.12,
            )

    def ee_position(self) -> np.ndarray:
        return np.asarray(
            self.pb.getLinkState(self.body, self.ee_link)[0],
            dtype=float,
        )
