#!/usr/bin/env python3
# This script solves an inverse kinematics problem using CASADI/IPOPT or QP to estimate joint angles from LSTM-augmented anatomical marker data.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat

from comfi_examples.urdf_utils import mks_registration, scale_human_model, Robot
from comfi_examples.utils import read_mks_data, read_subject_yaml
from comfi_examples.ik_utils import RT_IK

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

# Markers to skip during tracking/RMSE computation
MKS_TO_SKIP = [
    "LForearm",
    "LUArm",
    "RUArm",
    "RHJC_study",
    "LHJC_study",
    "r_pelvis",
    "l_pelvis",
    "LHand",
    "LHL2",
    "LHM5",
    "RForearm",
    "RHand",
    "RHL2",
    "RHM5",
    "L_sh1_study",
    "L_thigh1_study",
    "r_sh1_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "L_sh2_study",
    "L_thigh2_study",
    "r_sh2_study",
    "L_sh3_study",
    "L_thigh3_study",
    "r_sh3_study",
    "r_thigh3_study",
]

# Keys from keypoints to add to markers to get head pose
KEYS_TO_ADD = ["Nose", "Head", "Right_Ear", "Left_Ear", "Right_Eye", "Left_Eye"]

# Markers to track in IK
KEYS_TO_TRACK = [
    "Nose",
    "Head",
    "Right_Ear",
    "Left_Ear",
    "Right_Eye",
    "Left_Eye",
    "C7",
    "RASI",
    "LASI",
    "RPSI",
    "LPSI",
    "RSHO",
    "RELB",
    "RMELB",
    "RWRI",
    "RMWRI",
    "RANK",
    "RMANK",
    "RTOE",
    "R5MHD",
    "RHEE",
    "RKNE",
    "RMKNE",
    "LSHO",
    "LELB",
    "LMELB",
    "LWRI",
    "LMWRI",
    "LANK",
    "LMANK",
    "LTOE",
    "L5MHD",
    "LHEE",
    "LKNE",
    "LMKNE",
]

# Joints to lock during IK
JOINTS_TO_LOCK = [
    "middle_thoracic_X",
    "middle_thoracic_Y",
    "middle_thoracic_Z",
    "left_wrist_X",
    "left_wrist_Z",
    "right_wrist_X",
    "right_wrist_Z",
]

# Joint angle names for output CSV
JOINT_ANGLES_NAMES = [
    "Freeflyer_X[m]",
    "Freeflyer_Y[m]",
    "Freeflyer_Z[m]",
    "Freeflyer_quaternion_X",
    "Freeflyer_quaternion_Y",
    "Freeflyer_quaternion_Z",
    "Freeflyer_quaternion_W",
    "Left_Hip_Flexion_Extension[rad]",
    "Left_Hip_Abduction_Adduction[rad]",
    "Left_Hip_Internal_External_Rotation[rad]",
    "Left_Knee_Flexion_Extension[rad]",
    "Left_Ankle_Plantarflexion_Dorsiflexion[rad]",
    "Left_Ankle_Inversion_Eversion[rad]",
    "Lumbar_Flexion_Extension[rad]",
    "Lumbar_Lateral_Bending[rad]",
    "Left_Clavicle_Elevation_Depression[rad]",
    "Left_Shoulder_Flexion_Extension[rad]",
    "Left_Shoulder_Abduction_Adduction[rad]",
    "Left_Shoulder_Internal_External_Rotation[rad]",
    "Left_Elbow_Flexion_Extension[rad]",
    "Left_Elbow_Pronation_Supination[rad]",
    "Cervical_Flexion_Extension[rad]",
    "Cervical_Lateral_Bending[rad]",
    "Cervical_Internal_External_Rotation[rad]",
    "Right_Clavicle_Elevation_Depression[rad]",
    "Right_Shoulder_Flexion_Extension[rad]",
    "Right_Shoulder_Abduction_Adduction[rad]",
    "Right_Shoulder_Internal_External_Rotation[rad]",
    "Right_Elbow_Flexion_Extension[rad]",
    "Right_Elbow_Pronation_Supination[rad]",
    "Right_Hip_Flexion_Extension[rad]",
    "Right_Hip_Abduction_Adduction[rad]",
    "Right_Hip_Internal_External_Rotation[rad]",
    "Right_Knee_Flexion_Extension[rad]",
    "Right_Ankle_Plantarflexion_Dorsiflexion[rad]",
    "Right_Ankle_Inversion_Eversion[rad]",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Solve inverse kinematics using CASADI/IPOPT or QP from augmented marker data"
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
        "--dt",
        type=float,
        default=0.025,
        help="Time step for IK solver (1/fps). Default: 0.025 (40Hz)",
    )
    p.add_argument(
        "--ik-solver",
        choices=["ipopt", "qp"],
        default="ipopt",
        help="IK solver to use: 'ipopt' (CASADI/IPOPT) or 'qp' (Quadratic Programming). Default: ipopt",
    )
    p.add_argument(
        "--display",
        action="store_true",
        help="Enable real-time visualization in Meshcat during IK solving.",
    )
    p.add_argument(
        "--output-markers-file",
        default="markers_model_trajectories.csv",
        help="Output filename for estimated marker trajectories. Default: markers_model_trajectories.csv",
    )
    p.add_argument(
        "--output-angles-file",
        default="joint_angles.csv",
        help="Output filename for estimated joint angles. Default: joint_angles.csv",
    )
    return p.parse_args()


def validate_inputs(subject_id, task):
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"Unknown subject ID: {subject_id}")
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")


def process_ik(
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
    dt: float,
    ik_solver: str,
    display: bool,
    output_markers_file: str,
    output_angles_file: str,
) -> bool:
    """
    Process inverse kinematics for a single subject/task:
    - Load metadata and marker data
    - Scale and register the human model
    - Lock specified joints
    - Solve IK frame by frame
    - Save results and compute RMSE
    """
    metadata_yaml = comfi_root / "metadata" / f"{subject_id}.yaml"
    path_to_csv = output_root / subject_id / task / augmented_file
    path_to_kpt = output_root / subject_id / task / keypoints_file
    urdf_path = model_dir / urdf_file
    output_markers_csv = output_root / subject_id / task / output_markers_file
    output_angles_csv = output_root / subject_id / task / output_angles_file

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
    print(
        f"[INFO] Subject {subject_id}: height={subject_height}m, mass={subject_mass}kg, gender={gender}"
    )

    # Load augmented markers and keypoints
    print("[INFO] Loading marker data...")
    data_markers_lstm = pd.read_csv(path_to_csv)
    keypoints = pd.read_csv(path_to_kpt) / 1000  # Convert mm to m

    # Add specific keypoints to markers
    columns_to_add = [
        col for col in keypoints.columns if any(key + "_" in col for key in KEYS_TO_ADD)
    ]
    mks_data = pd.concat(
        [data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1
    )

    # Read marker data
    result_markers, start_sample_dict = read_mks_data(
        mks_data, start_sample=start_sample
    )
    print(f"[INFO] Loaded {len(result_markers)} frames")

    # Load URDF
    print(f"[INFO] Loading URDF from {urdf_path}")
    human = Robot(str(urdf_path), str(model_dir), isFext=True)
    human_model = human.model
    human_data = human.data
    human_collision_model = human.collision_model
    human_visual_model = human.visual_model

    # Scale the model to data
    print(f"[INFO] Scaling human model (with_hand={with_hand})")
    human_model = scale_human_model(
        human_model,
        start_sample_dict,
        with_hand=with_hand,
        gender=gender,
        subject_height=subject_height,
    )
    print(f"[INFO] Model DOF (before locking): {human_model.nq}")

    # Register markers
    print("[INFO] Registering markers to model")
    human_model = mks_registration(human_model, start_sample_dict, with_hand=False)
    human_data = pin.Data(human_model)

    # Lock joints
    print(f"[INFO] Locking {len(JOINTS_TO_LOCK)} joints")
    joint_ids_to_lock = []
    for jn in JOINTS_TO_LOCK:
        if human_model.existJointName(jn):
            joint_ids_to_lock.append(human_model.getJointId(jn))
        else:
            print(f"[WARNING] Joint {jn} does not belong to the model!")

    q0 = pin.neutral(human_model)
    human_model, human_visual_model = pin.buildReducedModel(
        human_model, human_visual_model, joint_ids_to_lock, q0
    )
    print(f"[INFO] Model DOF (after locking): {human_model.nq}")
    human_data = pin.Data(human_model)

    # Visualization setup (if enabled)
    viz = None
    viewer = None
    if display:
        print(f"[INFO] Initializing Meshcat visualizer at {meshcat_url}")

        # Override materials for the visual model
        for go in human_visual_model.geometryObjects:
            go.overrideMaterial = True
            go.meshColor = np.array([0.0, 1.0, 0.0, 0.7])  # Green with transparency

        # Create the visualizer
        viz = MeshcatVisualizer(human_model, human_collision_model, human_visual_model)
        viewer = meshcat.Visualizer(zmq_url=meshcat_url)
        viz.initViewer(viewer=viewer, open=True)
        viz.viewer.delete()  # Clear scene
        viz.loadViewerModel("human_model")

        # Add ground truth markers (RED)
        for marker_name in KEYS_TO_TRACK:
            viewer[f"gt_markers/{marker_name}"].set_object(
                meshcat.geometry.Sphere(0.02),
                meshcat.geometry.MeshLambertMaterial(color=0xFF0000, opacity=0.9),
            )

        # Add model markers (GREEN)
        for marker_name in KEYS_TO_TRACK:
            viewer[f"model_markers/{marker_name}"].set_object(
                meshcat.geometry.Sphere(0.015),  # Slightly smaller
                meshcat.geometry.MeshLambertMaterial(color=0x00FF00, opacity=0.8),
            )

        print("[INFO] Visualization setup complete")
    else:
        print("[INFO] Real-time visualization disabled")

    # Initialize IK solver
    print(f"[INFO] Initializing IK solver (method: {ik_solver.upper()})...")
    q = pin.neutral(human_model)

    # Set marker weights (all equal for now)
    omega = {key: 1.0 for key in KEYS_TO_TRACK}

    ik_class = RT_IK(human_model, start_sample_dict, q, KEYS_TO_TRACK, dt, omega)

    # Warm start with IPOPT (always done regardless of solver choice)
    print("[INFO] Warm starting with CASADI/IPOPT...")
    q = ik_class.solve_ik_sample_casadi()
    ik_class._q0 = q

    # Select solver method for the main loop
    if ik_solver == "ipopt":
        solve_method = ik_class.solve_ik_sample_casadi
        print("[INFO] Using CASADI/IPOPT solver for main IK loop")
    elif ik_solver == "qp":
        solve_method = ik_class.solve_ik_sample_quadprog
        print("[INFO] Using Quadratic Programming (QP) solver for main IK loop")
    else:
        raise ValueError(f"Unknown IK solver: {ik_solver}")

    # Main IK loop
    print(f"[INFO] Solving IK for {len(result_markers)} frames...")
    rmse_per_marker = {}
    q_list = []
    M_model_list = []

    for ii in range(start_sample, len(result_markers)):
        mks_dict = result_markers[ii]
        ik_class._dict_m = mks_dict

        # Solve IK using selected method
        q = solve_method()

        # Forward kinematics
        pin.forwardKinematics(human_model, human_data, q)
        pin.updateFramePlacements(human_model, human_data)

        M_model_frame = {}

        # Process each marker
        for marker in result_markers[ii].keys():
            if marker in MKS_TO_SKIP:
                continue

            # Ground truth position
            pos_gt = np.array(result_markers[ii][marker])

            # Model position
            M_model = human_data.oMf[human_model.getFrameId(marker)]
            pos_model = np.array(M_model.translation).flatten()

            # Store for CSV output
            M_model_frame[f"{marker}_x"] = M_model.translation[0]
            M_model_frame[f"{marker}_y"] = M_model.translation[1]
            M_model_frame[f"{marker}_z"] = M_model.translation[2]

            # Update visualization (if enabled)
            if display and viewer and marker in KEYS_TO_TRACK:
                transform_gt = meshcat.transformations.translation_matrix(pos_gt)
                viewer[f"gt_markers/{marker}"].set_transform(transform_gt)

                transform_model = meshcat.transformations.translation_matrix(pos_model)
                viewer[f"model_markers/{marker}"].set_transform(transform_model)

            # RMSE calculation
            sq_error = np.sum((pos_gt - pos_model) ** 2)
            if marker not in rmse_per_marker:
                rmse_per_marker[marker] = []
            rmse_per_marker[marker].append(sq_error)

        M_model_list.append(M_model_frame)

        # Update human model pose visualization
        if display and viz:
            viz.display(q)

        # Update for next iteration
        ik_class._q0 = q
        q_list.append(q)

        if ii % 10 == 0:
            print(f"[PROGRESS] Frame {ii}/{len(result_markers)}")

    print("[INFO] IK calculations complete")

    # Save estimated marker trajectories
    print(f"[INFO] Saving marker trajectories to {output_markers_csv}")
    df_markers = pd.DataFrame(M_model_list)
    output_markers_csv.parent.mkdir(parents=True, exist_ok=True)
    df_markers.to_csv(output_markers_csv, index=False)

    # Save joint angles
    print(f"[INFO] Saving joint angles to {output_angles_csv}")
    num_values = len(q_list[0])
    if len(JOINT_ANGLES_NAMES) != num_values:
        raise ValueError(
            f"JOINT_ANGLES_NAMES has {len(JOINT_ANGLES_NAMES)} entries but q has {num_values} DOFs."
        )
    df_angles = pd.DataFrame(q_list, columns=JOINT_ANGLES_NAMES)
    df_angles.to_csv(output_angles_csv, index=False)

    # Compute and display RMSE
    print("\n" + "=" * 60)
    print("TRACKING ERROR - Per-marker RMSE (meters):")
    print("=" * 60)
    rmse_global = 0
    nb_mks = 0
    for marker, sq_errors in sorted(rmse_per_marker.items()):
        nb_mks += 1
        rmse = np.sqrt(np.mean(sq_errors))
        print(f"  {marker:20s}: {rmse:.4f} m")
        rmse_global += rmse

    rmse_global = rmse_global / nb_mks
    print("=" * 60)
    print(f"GLOBAL RMSE (all markers): {rmse_global:.4f} m")
    print("=" * 60 + "\n")

    print(f"[SUCCESS] {subject_id}/{task} IK processing complete")
    return True


def main():
    args = parse_args()
    validate_inputs(args.subject_id, args.task)

    comfi_root = Path(args.comfi_root).resolve()
    output_root = Path(args.output_root).resolve()
    model_dir = Path(args.model_dir).resolve()

    try:
        process_ik(
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
            dt=args.dt,
            ik_solver=args.ik_solver,
            display=args.display,
            output_markers_file=args.output_markers_file,
            output_angles_file=args.output_angles_file,
        )
        print("[DONE] IK processing completed successfully")
    except Exception as e:
        print(f"[ERROR] IK processing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
