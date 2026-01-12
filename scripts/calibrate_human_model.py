#!/usr/bin/env python3
# This script allows calibrating a human model using augmented markers from the LSTM.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat

# Add the src folder to sys.path so that viewer modules can be found.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))

from comfi_examples.urdf_utils import *
from comfi_examples.utils import read_mks_data, read_subject_yaml
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

# Keys to add from keypoints to markers
KEYS_TO_ADD = ['Nose', 'Head', 'Right_Ear', 'Left_Ear', 'Right_Eye', 'Left_Eye']

# Joints to lock during visualization
JOINTS_TO_LOCK = [
    "middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z",
    "left_wrist_X", "left_wrist_Z", "right_wrist_X", "right_wrist_Z"
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate a human model using augmented markers from LSTM"
    )
    p.add_argument(
        "--id",
        dest="subject_id",
        required=True,
        help="Subject ID, e.g., --id 1012",
    )
    p.add_argument(
        "--task",
        required=True,
        help="Task name, e.g., --task RobotWelding",
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
        help="Path to output directory root (where augmented markers are).",
    )
    p.add_argument(
        "--model-dir",
        default=Path("model"),
        type=Path,
        help="Path to model directory containing URDF and meshes.",
    )
    p.add_argument(
        "--urdf-file",
        default="urdf/human.urdf",
        help="Relative path to URDF file inside model-dir. Default: urdf/human.urdf",
    )
    p.add_argument(
        "--augmented-file",
        default="augmented_markers.csv",
        help="Name of augmented markers CSV file. Default: augmented_markers.csv",
    )
    p.add_argument(
        "--keypoints-file",
        default="3d_keypoints_4cams.csv",
        help="Name of 3D keypoints CSV file. Default: 3d_keypoints_4cams.csv",
    )
    p.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Start sample for marker data. Default: 0",
    )
    p.add_argument(
        "--with-hand",
        action="store_true",
        help="Include hand scaling in model calibration.",
    )
    p.add_argument(
        "--meshcat-url",
        default="tcp://127.0.0.1:6000",
        help="Meshcat server URL. Default: tcp://127.0.0.1:6000",
    )
    p.add_argument(
        "--marker-radius",
        type=float,
        default=0.02,
        help="Marker sphere radius for visualization. Default: 0.02",
    )
    return p.parse_args()


def validate_inputs(subject_id, task):
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"Unknown subject ID: {subject_id}")
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")


def calibrate_human_model(
    subject_id: str,
    task: str,
    comfi_root: Path,
    output_root: Path,
    model_dir: Path,
    urdf_file: str,
    augmented_file: str,
    keypoints_file: str,
    start_sample: int,
    with_hand: bool,
    meshcat_url: str,
    marker_radius: float,
) -> bool:
    """
    Process calibration for a single subject/task:
    - Load metadata
    - Load augmented markers and keypoints
    - Scale and register the human model
    - Visualize in Meshcat
    """
    # Construct file paths
    metadata_yaml = comfi_root / "metadata" / f"{subject_id}.yaml"
    path_to_csv = output_root / subject_id / task / augmented_file
    path_to_kpt = output_root / subject_id / task / keypoints_file
    urdf_path = model_dir / urdf_file

    # Existence checks
    if not metadata_yaml.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_yaml}")
    if not path_to_csv.exists():
        raise FileNotFoundError(f"Missing augmented markers file: {path_to_csv}")
    if not path_to_kpt.exists():
        raise FileNotFoundError(f"Missing 3D keypoints file: {path_to_kpt}")
    if not urdf_path.exists():
        raise FileNotFoundError(f"Missing URDF file: {urdf_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Missing model directory: {model_dir}")

    # Load subject metadata
    _, subject_height, subject_mass, gender = read_subject_yaml(str(metadata_yaml))
    print(f"[INFO] Subject {subject_id}: height={subject_height}m, mass={subject_mass}kg, gender={gender}")

    # Load augmented markers and keypoints
    data_markers_lstm = pd.read_csv(path_to_csv)
    keypoints = pd.read_csv(path_to_kpt) / 1000  # Convert mm to m

    # Add specific keypoints to markers
    columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in KEYS_TO_ADD)]

    if len(data_markers_lstm) != len(keypoints):
        raise ValueError("Row count mismatch between augmented markers and keypoints")

    mks_data = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

    # Read marker data
    result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample, converter=1.0)
    mks_names = list(start_sample_dict.keys())
    print(f"[INFO] Loaded {len(mks_names)} markers")

    # Load URDF
    print(f"[INFO] Loading URDF from {urdf_path}")
    human = Robot(str(urdf_path), str(model_dir), isFext=True)
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    # Scale the model to data
    print(f"[INFO] Scaling human model (with_hand={with_hand})")
    human_model = scale_human_model(human_model, start_sample_dict, with_hand=with_hand, 
                                    gender=gender, subject_height=subject_height)
    print(f"[INFO] Model DOF: {human_model.nq}")

    # Register markers
    print("[INFO] Registering markers to model")
    human_model = mks_registration(human_model, start_sample_dict, with_hand=False, 
                                   gender=gender, subject_height=subject_height)

    human_data = pin.Data(human_model)
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    # VISUALIZATION
    # Lock some joints
    q0 = pin.neutral(human_model)
    joint_ids_to_lock = [human_model.getJointId(jn) for jn in JOINTS_TO_LOCK 
                         if human_model.existJointName(jn)]

    model, (collision_model, visual_model) = pin.buildReducedModel(
        human_model, [human_collision_model, human_visual_model], joint_ids_to_lock, q0
    )
    data = pin.Data(model)

    # Override materials on the final visual_model
    for go in human_visual_model.geometryObjects:
        go.overrideMaterial = True
        go.meshColor = np.array([0.0, 1.0, 0.0, 0.7])  # Green with transparency

    # Create the visualizer with consistent, reduced models
    print(f"[INFO] Initializing Meshcat visualizer at {meshcat_url}")
    viz = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
    viewer = meshcat.Visualizer(zmq_url=meshcat_url)
    viz.initViewer(viewer=viewer, open=True)
    viz.viewer.delete()  # Clear scene if relaunching
    viz.loadViewerModel("ref")

    # Display markers after registration on the model
    add_markers_to_meshcat(
        viewer,
        {},
        marker_names=mks_names,
        radius=marker_radius,
        default_color=0xFF0000,  # Red
        opacity=0.9
    )

    q = pin.neutral(human_model)
    pin.forwardKinematics(human_model, human_data, pin.neutral(human_model))
    pin.updateFramePlacements(human_model, human_data)

    # Get model marker positions
    model_markers = {}
    for name in mks_names:
        if human_model.existFrame(name):
            fid = human_model.getFrameId(name)
            model_markers[name] = human_data.oMf[fid].translation.copy()

    model_markers_list = [model_markers]

    q = pin.neutral(human_model)
    viz.display(q)

    # Set marker positions in visualization
    set_markers_frame(
        viz.viewer,
        model_markers_list,
        t=0,
        marker_names=mks_names,
        unit_scale=1.0
    )

    ###display markers without registration (same names so can't display both in same time)
    # add_markers_to_meshcat(
    #         viewer,
    #         result_markers,
    #         marker_names=mks_names,
    #         radius=0.025,
    #         default_color=0xFF0000,
    #         opacity=0.95,
    #     )
    # set_markers_frame(
    #             viz.viewer, result_markers, 0, marker_names=mks_names, unit_scale=1.0
    #         )


    print(f"[SUCCESS] {subject_id}/{task} calibration complete and visualized")
    return True


def main():
    args = parse_args()
    validate_inputs(args.subject_id, args.task)
    
    comfi_root = Path(args.comfi_root).resolve()
    output_root = Path(args.output_root).resolve()
    model_dir = Path(args.model_dir).resolve()

    try:
        calibrate_human_model(
            subject_id=args.subject_id,
            task=args.task,
            comfi_root=comfi_root,
            output_root=output_root,
            model_dir=model_dir,
            urdf_file=args.urdf_file,
            augmented_file=args.augmented_file,
            keypoints_file=args.keypoints_file,
            start_sample=args.start_sample,
            with_hand=args.with_hand,
            meshcat_url=args.meshcat_url,
            marker_radius=args.marker_radius,
        )
        print("[DONE] Calibration process completed successfully")
    except Exception as e:
        print(f"[ERROR] Calibration failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())



