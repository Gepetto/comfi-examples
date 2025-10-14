#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np

from comfi_examples.cam_utils import load_camera_parameters
from comfi_examples.triangulation_utils import triangulate_points_adaptive
from comfi_examples.utils import (
    read_mmpose_file, save_to_csv, load_transformation,
    transform_keypoints_list_cam0_to_mocap, read_mmpose_scores
)
from comfi_examples.linear_algebra_utils import butterworth_filter

SUBJECT_IDS = [
    "1012","1118","1508","1602","1847","2112","2198","2307","3361",
    "4162","4216","4279","4509","4612","4665","4687","4801","4827"
]
DS_TASKS = [
    "Screwing","ScrewingSat","Crouching","Picking","Hammering","HammeringSat","Jumping","Lifting",
    "QuickLifting","Lower","SideOverhead","FrontOverhead","RobotPolishing","RobotWelding",
    "Polishing","PolishingSat","SitToStand","Squatting","Static","Upper","CircularWalking",
    "StraightWalking","Welding","WeldingSat"
]

num_keypoints = 26
markers = [
    "Nose","Left_Eye","Right_Eye","Left_Ear","Right_Ear",
    "Left_Shoulder","Right_Shoulder","Left_Elbow","Right_Elbow",
    "Left_Wrist","Right_Wrist","Left_Hip","Right_Hip",
    "Left_Knee","Right_Knee","Left_Ankle","Right_Ankle","Head",
    "Neck","Mid_Hip","Left_Big_Toe","Right_Big_Toe","Left_Small_Toe","Right_Small_Toe","Left_Heel","Right_Heel"
]
header = [f"{m}_{axis}[mm]" for m in markers for axis in ("X","Y","Z")]

DEFAULT_CAM_IDS = [0, 2, 4, 6]

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the triangulation using keypoints extracted from multiple camera views"
    )
    p.add_argument("--id", dest="subject_ids", nargs="+", required=True,
                   help="Subject IDs (space-separated), e.g., --id 1012 1118")
    p.add_argument("--task", dest="tasks", nargs="+", required=True,
                   help="Task names (space-separated), e.g., --task RobotWelding Lifting")
    p.add_argument("--comfi-root", required=True,
                   help="Path to COMFI dataset root.")
    p.add_argument("--cams", dest="cam_ids", nargs="+", type=int, choices=DEFAULT_CAM_IDS,
                   default=DEFAULT_CAM_IDS,
                   help="Camera IDs to use for the triangulation. Default: 0 2 4 6")
    return p.parse_args()

def validate_lists(subject_ids, tasks):
    unknown_ids = sorted(set(subject_ids) - set(SUBJECT_IDS))
    unknown_tasks = sorted(set(tasks) - set(DS_TASKS))
    msgs = []
    if unknown_ids:
        msgs.append(f"Unknown subject IDs: {', '.join(unknown_ids)}")
    if unknown_tasks:
        msgs.append(f"Unknown tasks: {', '.join(unknown_tasks)}")
    if msgs:
        raise ValueError("; ".join(msgs))

def process_one(comfi_root: Path, sid: str, task: str, cam_ids: list[int]) -> bool:
    if len(cam_ids) < 2:
        raise ValueError(f"At least two cameras are required for triangulation; got {cam_ids}")

    keypoints_dir = Path("output").resolve() / "res_hpe" / sid / task
    file_paths = [keypoints_dir / f"keypoints_cam{c}.csv" for c in cam_ids]

    transform_file = comfi_root / "cam_params" / sid / "extrinsics" / "cam_to_world" / f"camera_{cam_ids[0]}" / "soder.txt"
    intrinsics_dir = comfi_root / "cam_params" / sid / "intrinsics"
    extrinsics_dir = comfi_root / "cam_params" / sid / "extrinsics"

    # Output:
    out_dir = Path("output").resolve() / "res_hpe" / sid / task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"3d_keypoints_{len(cam_ids)}cams.csv"

    # ---- Existence checks; fail fast with clear messages ----
    missing = [str(p) for p in file_paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing 2D keypoints CSV(s):\n  " + "\n  ".join(missing))
    if not transform_file.exists():
        raise FileNotFoundError(f"Missing transform file (cam0->mocap): {transform_file}")
    if not intrinsics_dir.exists():
        raise FileNotFoundError(f"Missing calibration directory: {intrinsics_dir}")
    if not extrinsics_dir.exists():
        raise FileNotFoundError(f"Missing calibration directory: {extrinsics_dir}")

    # ---- Load transforms & calibration ----
    R_trans, d_trans, _, _ = load_transformation(str(transform_file))

    # Note: load_camera_parameters should raise if any per-camera calib file is missing.
    mtxs, dists, projections, rotations, translations = load_camera_parameters(
        str(intrinsics_dir), str(extrinsics_dir), camera_ids=cam_ids
    )

    # ---- Load 2D detections + scores (only for the selected cams) ----
    camera_data = [read_mmpose_file(str(fp)) for fp in file_paths]
    # camera_data[k] is list-like per frame; flatten into (F, K, 2) with K=num_keypoints
    uvs = [
        np.array([[line[2 * i], line[2 * i + 1]] for line in data for i in range(num_keypoints)])
          .reshape(-1, num_keypoints, 2)
        for data in camera_data
    ]
    scores = read_mmpose_scores([str(fp) for fp in file_paths])

    # ---- Triangulation ----
    threshold = 0.0 if len(cam_ids) == 2 else 0.5
    keypoints_in_cam0_list = triangulate_points_adaptive(
        uvs, mtxs, dists, projections, scores, threshold
    )

    # ---- Transform to mocap/world frame (using cam0->mocap) ----
    keypoints_in_mocap = transform_keypoints_list_cam0_to_mocap(
        keypoints_in_cam0_list, R_trans, d_trans
    )

    # ---- Filtering (40 Hz default; adjust if needed) ----
    filtered = butterworth_filter(
        data=keypoints_in_mocap,
        cutoff_frequency=10.0,
        order=5,
        sampling_frequency=40
    )

    # ---- Save ----
    save_to_csv(filtered, str(out_csv), header=header)
    print(f"[SAVED] {sid}/{task} ({len(cam_ids)} cams) → {out_csv}")
    return True

def main():
    args = parse_args()
    validate_lists(args.subject_ids, args.tasks)
    comfi_root = Path(args.comfi_root).resolve()

    total = 0
    ok = 0
    for sid in args.subject_ids:
        for task in args.tasks:
            total += 1
            ok += int(process_one(comfi_root, sid, task, args.cam_ids))

    print(f"[DONE] {ok}/{total} combinations processed.")

if __name__ == "__main__":
    main()
