#!/usr/bin/env python3
import argparse
from pathlib import Path
import os
import numpy as np
import pandas as pd

from comfi_examples.utils  import read_mks_data
from comfi_examples.utils import kabsch_global, plot_aligned_markers,compute_mpjpe

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
        description="Run the triangulation using keypoints extracted from multiple camera views"
    )
    p.add_argument("--id", dest="subject_ids", nargs="+", required=True,
                   help="Subject IDs (space-separated), e.g., --id 1012 1118")
    p.add_argument("--task", dest="tasks", nargs="+", required=True,
                   help="Task names (space-separated), e.g., --task RobotWelding Lifting")
    p.add_argument("--comfi-root", default=Path(os.environ.get("COMFI_ROOT", "COMFI")),
                   help="Path to COMFI dataset root.")
    p.add_argument("--nb-cams", default=4,
                   help="Number of cameras used for the triangulation of the hpe keypoints")
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

def get_marker_array(mks_list, marker_name):
    """
    mks_list: list of dicts with keys 'name' and 'value'
    marker_name: str
    returns: array (T,3)
    """
    for mk in mks_list:
        if mk["name"] == marker_name:
            return mk["value"]
    raise ValueError(f"Marker {marker_name} not found")

def process_one(comfi_root, sid, task, nb_cams):

    path_to_jcp_mocap= comfi_root / "mocap" / "aligned" / sid / task / "joint_center_positions.csv" #data from mocap and hpe should be with same unit
    if not path_to_jcp_mocap.exists():
        raise FileNotFoundError(f"Missing joint center positions mocap: {path_to_jcp_mocap}")

    df_mocap = pd.read_csv(path_to_jcp_mocap)
    mks_mocap, start_sample_mocap = read_mks_data(df_mocap, start_sample=0)

    path_to_jcp_hpe = Path("output").resolve() / "res_hpe" / sid / task / f"3d_keypoints_{nb_cams}cams.csv"
    df_hpe = pd.read_csv(path_to_jcp_hpe)
    mks_hpe, start_sample_hpe = read_mks_data(df_hpe, start_sample=0)
    # print(mks_mocap)
    # print(mks_mocap[0])

    #take only common markers
    common_mks = sorted(set(start_sample_mocap.keys()) & set(start_sample_mocap.keys()))
    print("Common markers:", common_mks)

    T = len(mks_mocap)
    N = len(common_mks)

    P_mocap_seq = np.zeros((T, N, 3))
    P_hpe_seq   = np.zeros((T, N, 3))

    for i, mk in enumerate(common_mks):
        P_mocap_seq[:, i, :] = np.stack([frame[mk] for frame in mks_mocap], axis=0)
        P_hpe_seq[:, i, :]   = np.stack([frame[mk] for frame in mks_hpe], axis=0)

    R, t, rms = kabsch_global(P_hpe_seq, P_mocap_seq)
    print("RMS alignment error (mm):", rms)

    P_hpe_seq_aligned = (P_hpe_seq @ R.T) + t  # (T,N,3)
    plot_aligned_markers(P_mocap_seq, P_hpe_seq_aligned, common_mks)
    mpjpe = compute_mpjpe(P_hpe_seq_aligned, P_mocap_seq)
    print("MPJPE (mm):", mpjpe)

    T, N, _ = P_hpe_seq_aligned.shape
    df_hpe_aligned = pd.DataFrame(P_hpe_seq_aligned.reshape(T, 3*N),
                                columns=df_mocap.columns,
                                index=df_mocap.index)

    output_csv =  Path("output").resolve() / "res_hpe"/ sid / task / "3d_keypoints_aligned.csv"
    df_hpe_aligned.to_csv(output_csv, index=False)
    print("Aligned HPE saved to:", output_csv)
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
            ok += int(process_one(comfi_root, sid, task, args.nb_cams))

    print(f"[DONE] {ok}/{total} combinations processed.")

if __name__ == "__main__":
    main()
