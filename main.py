from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

from rover_sim.config import CASE_LABELS, MISSION_LABELS, PATH_LABELS, SCENARIO_LABELS, SimConfig
from rover_sim.simulation import run_simulation


def parse_args():
    ap = argparse.ArgumentParser(description="Six-wheel Mars rover research simulation")
    ap.add_argument("--case", choices=list(CASE_LABELS), default="C3")
    ap.add_argument("--scenario", choices=list(SCENARIO_LABELS), default="nominal")
    ap.add_argument("--path", dest="path_name", choices=list(PATH_LABELS), default="s_curve")
    ap.add_argument("--mission", choices=list(MISSION_LABELS), default="path_follow")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--viewer",
        choices=["native", "clean", "bullet", "headless"],
        default="native",
        help="native=fast OpenGL; clean=custom Tk; bullet=raw Bullet GUI; headless=no window",
    )
    ap.add_argument("--headless", action="store_true", help="Alias for --viewer headless")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--arm-demo", action="store_true", help="Enable arm retrieval after reaching the goal")
    ap.add_argument("--no-camera-follow", action="store_true")
    ap.add_argument("--show-debug-ui", action="store_true", help="Only applies to --viewer bullet")
    ap.add_argument("--auto-close", action="store_true", help="Close the 3D window automatically when the run finishes")
    ap.add_argument("--duration", type=float, default=0.0, help="Maximum simulation time [s]; 0 = automatic")
    ap.add_argument("--speed", type=float, default=None, help="Override desired speed [m/s]")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--video", type=str, default=None, help="Raw Bullet GUI MP4 logging only")

    # Random-sample mission overrides. If omitted, config.py values are used.
    ap.add_argument("--drop-time", type=float, default=None, help="Random-sample mission: object drop time [s]")
    ap.add_argument("--drop-height", type=float, default=None, help="Random-sample mission: drop height above terrain [m]")
    ap.add_argument("--planner-max-slope", type=float, default=None, help="A*: maximum traversable slope [deg]")
    ap.add_argument("--planner-resolution", type=float, default=None, help="A*: grid spacing [m]")
    return ap.parse_args()


def main():
    faulthandler.enable()
    args = parse_args()

    cfg = SimConfig()
    cfg.case = args.case
    cfg.scenario = args.scenario
    cfg.path_name = args.path_name
    cfg.mission = args.mission
    cfg.seed = args.seed
    cfg.terrain.seed = args.seed
    cfg.viewer = "headless" if args.headless else args.viewer
    cfg.gui = cfg.viewer != "headless"
    cfg.realtime = not args.no_realtime

    # random_sample always performs retrieval, so the arm is enabled automatically.
    cfg.arm_enabled = bool(args.arm_demo or args.mission == "random_sample")
    cfg.camera_follow = not args.no_camera_follow
    cfg.clean_gui = not args.show_debug_ui
    cfg.keep_gui_open = not args.auto_close
    cfg.duration = args.duration

    if args.speed is not None:
        cfg.control.desired_speed = args.speed
    if args.drop_time is not None:
        cfg.mission_cfg.drop_time_s = max(0.0, args.drop_time)
    if args.drop_height is not None:
        cfg.mission_cfg.drop_height_m = max(0.3, args.drop_height)
    if args.planner_max_slope is not None:
        cfg.mission_cfg.planner_max_slope_deg = max(3.0, args.planner_max_slope)
    if args.planner_resolution is not None:
        cfg.mission_cfg.planner_resolution_m = max(0.15, args.planner_resolution)

    default_name = f"{args.case}_{args.scenario}_{args.mission}_{args.path_name}_seed{args.seed}"
    cfg.output_dir = args.output or Path("results") / default_name
    cfg.video_path = args.video

    try:
        run_simulation(cfg)
    except Exception:
        report = traceback.format_exc()
        print("\n[FATAL] Simulation stopped because of a Python exception:\n", file=sys.stderr)
        print(report, file=sys.stderr, flush=True)
        try:
            cfg.output_dir.mkdir(parents=True, exist_ok=True)
            (cfg.output_dir / "crash_report.txt").write_text(report, encoding="utf-8")
            print(f"Crash report saved to: {(cfg.output_dir / 'crash_report.txt').resolve()}", file=sys.stderr)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
