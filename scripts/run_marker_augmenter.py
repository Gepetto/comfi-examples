#!/usr/bin/env python3
# This script uses an LSTM model from OpenCap to augment 26 keypoint data to 43 anatomical markers.

import os
import sys
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd

# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src"))
)

from comfi_examples.augmenter_utils import augmentTRC, loadModel
from comfi_examples.utils import save_to_csv, read_subject_yaml
from comfi_examples.linear_algebra_utils import butterworth_filter

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

TASKS = [
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

MARKERS = [
    "RASI",
    "LASI",
    "RPSI",
    "LPSI",
    "RKNE",
    "RMKNE",
    "RANK",
    "RMANK",
    "RTOE",
    "R5MHD",
    "RHEE",
    "LKNE",
    "LMKNE",
    "LANK",
    "LMANK",
    "LTOE",
    "LHEE",
    "L5MHD",
    "RSHO",
    "LSHO",
    "C7",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
    "RELB",
    "RMELB",
    "RWRI",
    "RMWRI",
    "LELB",
    "LMELB",
    "LWRI",
    "LMWRI",
]

HEADER = [
    f"{marker}_{axis}" for marker in MARKERS for axis in ("X[mm]", "Y[mm]", "Z[mm]")
]

BUFFER_SIZE = 30


def parse_args():
    p = argparse.ArgumentParser(
        description="Augment 26 keypoints to 43 anatomical markers using LSTM model"
    )
    p.add_argument(
        "--id",
        dest="subject_ids",
        nargs="+",
        required=True,
        help="Subject IDs (space-separated), e.g., --id 1012 1118",
    )
    p.add_argument(
        "--task",
        dest="tasks",
        nargs="+",
        required=True,
        help="Task names (space-separated), e.g., --task RobotWelding Lifting",
    )
    p.add_argument(
        "--comfi-root",
        default=Path(os.environ.get("COMFI_ROOT", "COMFI")),
        type=Path,
        help="Path to COMFI dataset root.",
    )
    p.add_argument(
        "--output-root",
        default=Path("output").resolve() / "res_hpe",
        type=Path,
        help="Path to output directory root (where 3D keypoints are).",
    )
    p.add_argument(
        "--augmenter-path",
        default="./augmentation_model",
        help="Path to augmentation model directory.",
    )
    p.add_argument(
        "--input-file",
        default="3d_keypoints_4cams.csv",
        help="Name of input 3D keypoints file. Default: 3d_keypoints_4cams.csv",
    )
    p.add_argument(
        "--output-file",
        default="augmented_markers.csv",
        help="Name for output augmented markers file. Default: augmented_markers.csv",
    )
    p.add_argument(
        "--cutoff-freq",
        type=float,
        default=10.0,
        help="Butterworth filter cutoff frequency (Hz). Default: 10.0",
    )
    p.add_argument(
        "--filter-order",
        type=int,
        default=5,
        help="Butterworth filter order. Default: 5",
    )
    p.add_argument(
        "--sampling-freq",
        type=float,
        default=40.0,
        help="Sampling frequency (Hz). Default: 40.0",
    )
    return p.parse_args()


def validate_lists(subject_ids, tasks):
    unknown_ids = sorted(set(subject_ids) - set(SUBJECT_IDS))
    unknown_tasks = sorted(set(tasks) - set(TASKS))
    msgs = []
    if unknown_ids:
        msgs.append(f"Unknown subject IDs: {', '.join(unknown_ids)}")
    if unknown_tasks:
        msgs.append(f"Unknown tasks: {', '.join(unknown_tasks)}")
    if msgs:
        raise ValueError("; ".join(msgs))


def process_one(
    subject_id: str,
    task: str,
    output_root: Path,
    metadata_root: Path,
    augmenter_path: str,
    input_file: str,
    output_file: str,
    cutoff_freq: float,
    filter_order: int,
    sampling_freq: float,
    warmed_models,
) -> bool:
    """
    Process a single subject/task combination:
    - Load 3D keypoints
    - Augment using LSTM model
    - Filter results
    - Save to CSV
    """
    # Construct file paths
    input_csv = output_root / subject_id / task / input_file
    output_csv = output_root / subject_id / task / output_file
    metadata_yaml = metadata_root / f"{subject_id}.yaml"
    print(metadata_yaml)

    # Existence checks
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing 3D keypoints file: {input_csv}")
    if not metadata_yaml.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_yaml}")

    # Load subject metadata
    _, subject_height, subject_mass, _ = read_subject_yaml(str(metadata_yaml))

    # Load 3D keypoints
    data = pd.read_csv(input_csv).values / 1000
    num_columns = data.shape[1]

    if num_columns % 3 != 0:
        raise ValueError(
            f"Unexpected number of columns: {num_columns}. Should be divisible by 3."
        )

    num_keypoints = num_columns // 3
    coordinates_per_keypoint = 3

    # Initialize buffer and result list
    keypoints_buffer = deque(maxlen=BUFFER_SIZE)
    augmented_markers_list = []
    first_frame = True

    # Process each frame
    for i in range(len(data)):
        frame_data = data[i].reshape(num_keypoints, coordinates_per_keypoint)

        if first_frame:
            # Fill buffer with first frame
            for _ in range(BUFFER_SIZE):
                keypoints_buffer.append(np.array(frame_data))
            first_frame = False
        else:
            keypoints_buffer.append(np.array(frame_data))

        if len(keypoints_buffer) == BUFFER_SIZE:
            keypoints_buffer_array = np.array(keypoints_buffer)
            augmented_markers = augmentTRC(
                keypoints_buffer_array,
                subject_mass=subject_mass,
                subject_height=subject_height,
                models=warmed_models,
                augmenterDir=augmenter_path,
                augmenter_model="v0.3",
            )
            augmented_markers_list.append(augmented_markers)

    # Stack all augmented frames
    augmented_array = np.vstack(augmented_markers_list)

    # Apply Butterworth filter
    filtered_data = butterworth_filter(
        data=augmented_array,
        cutoff_frequency=cutoff_freq,
        order=filter_order,
        sampling_frequency=sampling_freq,
    )

    # Create output directory if needed
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    save_to_csv(filtered_data, str(output_csv), header=HEADER)
    print(f"[SAVED] {subject_id}/{task} → {output_csv}")
    return True


def main():
    args = parse_args()
    validate_lists(args.subject_ids, args.tasks)

    # Construire le chemin metadata depuis comfi_root
    comfi_root = Path(args.comfi_root).resolve()
    metadata_root = comfi_root / "metadata"

    # Load LSTM model once (expensive operation)
    print("[INFO] Loading LSTM model...")
    warmed_models = loadModel(
        augmenterDir=args.augmenter_path,
        augmenterModelName="LSTM",
        augmenter_model="v0.3",
    )
    print("[INFO] Model loaded successfully.")

    total = 0
    ok = 0
    for subject_id in args.subject_ids:
        for task in args.tasks:
            total += 1
            try:
                ok += int(
                    process_one(
                        subject_id=subject_id,
                        task=task,
                        output_root=args.output_root,
                        metadata_root=metadata_root,  # Utilise le chemin construit
                        augmenter_path=args.augmenter_path,
                        input_file=args.input_file,
                        output_file=args.output_file,
                        cutoff_freq=args.cutoff_freq,
                        filter_order=args.filter_order,
                        sampling_freq=args.sampling_freq,
                        warmed_models=warmed_models,
                    )
                )
            except Exception as e:
                print(f"[ERROR] {subject_id}/{task}: {e}")

    print(f"[DONE] {ok}/{total} combinations processed.")


if __name__ == "__main__":
    main()
