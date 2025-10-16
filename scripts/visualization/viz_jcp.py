#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import meshcat
from pinocchio.visualize import MeshcatVisualizer
import time
import imageio

from comfi_examples.utils import read_mks_data
from comfi_examples.viz_utils import add_markers_to_meshcat, set_markers_frame

SUBJECT_IDS = [
    "1012",
    "1118",
    "1508",
    "1602",
    "1847",
    "2112",
    "2198",
    "2307",
    "3361",
    "4162",
    "4216",
    "4279",
    "4509",
    "4612",
    "4665",
    "4687",
    "4801",
    "4827",
]
DS_TASKS = [
    "Screwing",
    "ScrewingSat",
    "Crouching",
    "Picking",
    "Hammering",
    "HammeringSat",
    "Jumping",
    "Lifting",
    "QuickLifting",
    "Lower",
    "SideOverhead",
    "FrontOverhead",
    "RobotPolishing",
    "RobotWelding",
    "Polishing",
    "PolishingSat",
    "SitToStand",
    "Squatting",
    "Static",
    "Upper",
    "CircularWalking",
    "StraightWalking",
    "Welding",
    "WeldingSat",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize COMFI joint_center_positions from either mocap or HPE in Meshcat."
    )

    p.add_argument(
        "--mode", choices=["mocap", "hpe"], required=True, help="Choose source type"
    )

    p.add_argument("--id", dest="subject_id", required=True, help="ID (e.g., 1012)")
    p.add_argument("--task", required=True, help="Task name (e.g., RobotWelding)")
    p.add_argument(
        "--freq",
        type=int,
        choices=[40, 100],
        required=True,
        help="Sampling frequency: 40 (aligned) or 100 (raw).",
    )
    p.add_argument(
        "--start", type=int, default=0, help="Start frame index (inclusive). Default: 0"
    )
    p.add_argument(
        "--stop",
        type=int,
        default=None,
        help="Stop frame index (exclusive). Default: None (till end)",
    )

    p.add_argument(
        "--comfi-root",
        default=Path(os.environ.get("COMFI_ROOT", "COMFI")),
        help="Path to comfi root directory (only for jcp_mocap)",
    )
    p.add_argument("--nb-cams", type=int, help="Number of cameras (only for jcp_hpe)")
    p.add_argument("--save-video", action="store_true", help="Save a video of the visualization")
    
    args = p.parse_args()

    if args.mode == "jcp_mocap":
        if not args.comfi_root:
            p.error("--comfi-root is required when --mode jcp_mocap")
        if args.nb_cams is not None:
            p.error("--nb-cams is not allowed when --mode jcp_mocap")

    if args.mode == "jcp_hpe":
        if args.comfi_root is not None:
            p.error("--comfi-root is not allowed when --mode jcp_hpe")
        if args.nb_cams is None:
            p.error("--nb-cams is required when --mode jcp_hpe")

    return p.parse_args()


def main():
    args = parse_args()

    # Minimal friendly validation
    if args.subject_id not in SUBJECT_IDS:
        raise ValueError(
            f"Unknown subject ID '{args.subject_id}'. Allowed: {', '.join(SUBJECT_IDS)}"
        )
    if args.task not in DS_TASKS:
        raise ValueError(f"Unknown task '{args.task}'. Allowed: {', '.join(DS_TASKS)}")

    split_folder = "aligned" if args.freq == 40 else "raw"

    if args.mode == "mocap":
        comfi_root = Path(args.comfi_root)
        csv_path = (
            comfi_root
            / "mocap"
            / split_folder
            / args.subject_id
            / args.task
            / "joint_center_positions.csv"
        )
    else:
        csv_path = (
            Path("output").resolve()
            / "res_hpe"
            / args.subject_id
            / args.task
            / f"3d_keypoints_{args.nb_cams}cams.csv"
        )

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: the task {args.task} is not available for id {args.subject_id}"
        )
    
    if args.save_video:
        video_path = (
            Path("output").resolve()
            / "res_hpe"
            / args.subject_id
            / args.task
            / f"viz_jcp.mp4"
        )

    # Load CSV
    df = pd.read_csv(csv_path)
    jcp_dict, start_sample_dict = read_mks_data(df, start_sample=0, converter=1000.0)
    mks_names = list(start_sample_dict.keys())

    # Frame range
    n = len(jcp_dict)
    start = max(0, args.start)
    stop = n if args.stop is None else min(args.stop, n)
    if start >= stop:
        raise ValueError(f"Invalid range: start={start}, stop={stop}, total={n}")

    # Viewer
    viewer = meshcat.Visualizer()
    viz = MeshcatVisualizer()
    viz.initViewer(viewer, open=True)
    viz.viewer.delete()
    native_viz = viz.viewer
    native_viz["/Background"].set_property("top_color", list((1, 1, 1)))
    native_viz["/Background"].set_property("bottom_color", list((1, 1, 1)))
    native_viz["/Grid"].set_transform(
        np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.0], [0, 0, 0, 1]])
    )

    # Markers
    add_markers_to_meshcat(
        viewer,
        jcp_dict,
        marker_names=mks_names,
        radius=0.025,
        default_color=0x00FF00,
        opacity=0.95,
    )

    # Animate
    if args.save_video:
        set_markers_frame(viewer, jcp_dict, start, marker_names=mks_names, unit_scale=1.0)
        input('Pause to set the view in Meshcat, press Enter to start the visualization')
        images=[]
        for i in range(start, stop):
            set_markers_frame(viewer, jcp_dict, i, marker_names=mks_names, unit_scale=1.0)
            images.append(viz.viewer.get_image())
            time.sleep(0.80 * (1 / args.freq))
        os.makedirs(video_path.parent, exist_ok=True)
        imageio.mimsave(video_path, images, fps=args.freq)
        print(
            f"[OK] Visualized {stop - start} frames | ID {args.subject_id} | Task {args.task} | {args.freq} Hz"
        )
        print(f"[SRC] {csv_path}")
        print(f"[VIDEO] Video saved to {video_path}") 
    else:
        for i in range(start, stop):
            set_markers_frame(viewer, jcp_dict, i, marker_names=mks_names, unit_scale=1.0)
            time.sleep(0.80 * (1 / args.freq))

        print(
            f"[OK] Visualized {stop - start} frames | ID {args.subject_id} | Task {args.task} | {args.freq} Hz"
        )
        print(f"[SRC] {csv_path}")


if __name__ == "__main__":
    main()
