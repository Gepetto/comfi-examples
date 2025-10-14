#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import os

from comfi_examples.utils import read_mks_data
from comfi_examples.urdf_utils import compute_joint_centers_from_mks

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

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute the joint center positions from markers_trajectories.csv or markers_model_trajectories.csv and store the result in file for reuse."
    )
    # Allow multiple IDs / tasks (space-separated)
    p.add_argument("--id", dest="subject_ids", nargs="+", required=True,
                   help="Subject IDs (space-separated), e.g., --id 1012 1118")
    p.add_argument("--task", dest="tasks", nargs="+", required=True,
                   help="Task names (space-separated), e.g., --task RobotWelding Lifting")
    p.add_argument("--comfi-root", default=Path(os.environ.get("COMFI_ROOT", "COMFI")),
                   help="Path to COMFI dataset root.")
    p.add_argument("--freq", type=int, choices=[40, 100], required=True,
                   help="Sampling frequency: 40 (aligned) or 100 (raw).")
    p.add_argument("--mkset", choices=["meas","est"], default="meas",
                   help="Markers type (ground-truth measured or model-estimated). Default: meas")
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

def mks_csv_path(comfi_root: Path, split_folder: str, subject_id: str, task: str, mkset: str) -> Path:
    filename = "markers_trajectories.csv" if mkset == "meas" else "markers_model_trajectories.csv"
    # Layout: <COMFI_ROOT>/mocap/<aligned|raw>/<id>/<task>/<filename>
    return comfi_root / "mocap" / split_folder / subject_id / task / filename

def process_one(comfi_root: Path, split_folder: str, subject_id: str, task: str, mkset: str, converter: float) -> bool:
    """Returns True if processed OK, False if skipped (e.g., file missing)."""
    src = mks_csv_path(comfi_root, split_folder, subject_id, task, mkset)
    if not src.exists():
        print(f"[SKIP] Missing markers CSV: {src}")
        return False

    try:
        mks_df = pd.read_csv(src)
    except Exception as e:
        print(f"[SKIP] Failed to read {src}: {e}")
        return False

    # If units are meters, set converter=1.0; if mm, set converter=1000.0
    mks_dict, mks_start_sample_dict = read_mks_data(
        mks_df, start_sample=0, converter=converter
    )

    jcp_rows = []
    for frame_id in range(len(mks_dict)):
        markers_frame = mks_dict[frame_id]
        jcp = compute_joint_centers_from_mks(markers_frame)

        flat = {}
        for name, coords in jcp.items():
            flat[f"{name}_X[mm]"] = coords[0] * converter
            flat[f"{name}_Y[mm]"] = coords[1] * converter
            flat[f"{name}_Z[mm]"] = coords[2] * converter
        jcp_rows.append(flat)

    jcp_df = pd.DataFrame(jcp_rows)

    out_dir = Path(f"output/mocap/{split_folder}/{subject_id}/{task}").resolve()
    os.makedirs(out_dir, exist_ok=True)
    out_csv = out_dir / f"joint_center_positions_{mkset}.csv"

    jcp_df.to_csv(out_csv, index=False)
    print(f"[SAVED] {subject_id}/{task} → {out_csv}")
    return True

def main():
    args = parse_args()
    # Set your unit converter here:
    converter = 1000.0  # if data already in meters, set to 1.0

    validate_lists(args.subject_ids, args.tasks)
    split_folder = "aligned" if args.freq == 40 else "raw"
    comfi_root = Path(args.comfi_root)

    total = 0
    ok = 0
    for sid in args.subject_ids:
        for task in args.tasks:
            total += 1
            ok += int(process_one(comfi_root, split_folder, sid, task, args.mkset, converter))

    print(f"[DONE] {ok}/{total} combinations processed.")

if __name__ == "__main__":
    main()
