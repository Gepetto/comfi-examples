#!/usr/bin/env python3
# This script compares joint angles from motion capture (ground truth) with IK estimates, computing RMSE, MAE, and correlation metrics.

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlation_lags, correlate
from comfi_examples.utils import read_specific_joint

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

# DOF names for comparison
DOFS = [
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

# Joints to exclude from comparison (e.g., locked joints)
EXCLUDED_JOINTS = [
    "Lwrist_flex_ext",
    "Lwrist_x",
    "Rwrist_flex_ext",
    "Rwrist_x",
    "Lelbow_pron_supi",
    "Relbow_pron_supi",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare joint angles from motion capture and IK estimation"
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
        help="Path to output directory root (where IK results are).",
    )
    p.add_argument(
        "--mocap-dir",
        default="mocap/aligned",
        help="Relative path to mocap data inside comfi-root. Default: mocap/aligned",
    )
    p.add_argument(
        "--ik-angles-file",
        default="joint_angles.csv",
        help="Name of IK joint angles CSV file. Default: joint_angles.csv",
    )
    p.add_argument(
        "--mocap-angles-file",
        default="joint_angles.csv",
        help="Name of mocap joint angles CSV file. Default: joint_angles.csv",
    )
    p.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Start sample for comparison. Default: 0",
    )
    p.add_argument(
        "--start-dof",
        type=int,
        default=0,
        help="Start DOF index for comparison (0=freeflyer, 7=first joint). Default: 0",
    )
    p.add_argument(
        "--sync-joint",
        default="Right_Knee_Flexion_Extension[rad]",
        help="Joint name to use for temporal synchronization. Default: Right_Knee_Flexion_Extension[rad]",
    )
    p.add_argument(
        "--plots-per-figure",
        type=int,
        default=6,
        help="Number of joint plots per figure. Default: 6",
    )
    p.add_argument(
        "--show-plots",
        action="store_true",
        help="Display plots interactively (blocks execution until closed).",
    )
    p.add_argument(
        "--save-plots",
        action="store_true",
        help="Save plots to output directory.",
    )
    p.add_argument(
        "--output-plot-dir",
        default="comparison_plots",
        help="Directory name for saved plots (inside output-root/subject/task/). Default: comparison_plots",
    )
    return p.parse_args()


def validate_inputs(subject_id, task):
    if subject_id not in SUBJECT_IDS:
        raise ValueError(f"Unknown subject ID: {subject_id}")
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")


def synchronize_signals(sig1, sig2):
    """
    Synchronize two signals by computing cross-correlation.

    Args:
        sig1: numpy array, reference signal
        sig2: numpy array, signal to be shifted

    Returns:
        lag: number of samples sig2 is shifted relative to sig1 (+ means sig2 delayed)
    """
    corr = correlate(sig1, sig2, mode="full")
    lags = correlation_lags(len(sig1), len(sig2), mode="full")
    lag = lags[np.argmax(corr)]
    return lag


def comparison_joint_angles(
    subject_id: str,
    task: str,
    comfi_root: Path,
    output_root: Path,
    mocap_dir: str,
    ik_angles_file: str,
    mocap_angles_file: str,
    start_sample: int,
    start_dof: int,
    sync_joint: str,
    plots_per_figure: int,
    show_plots: bool,
    save_plots: bool,
    output_plot_dir: str,
) -> bool:
    """
    Compare joint angles from mocap and IK:
    - Load both datasets
    - Synchronize temporally
    - Compute RMSE, MAE, and correlation
    - Generate comparison plots
    """
    path_mocap = comfi_root / mocap_dir / subject_id / task / mocap_angles_file
    path_ik = output_root / subject_id / task / ik_angles_file

    if not path_mocap.exists():
        raise FileNotFoundError(f"Missing mocap angles file: {path_mocap}")
    if not path_ik.exists():
        raise FileNotFoundError(f"Missing IK angles file: {path_ik}")

    print(f"[INFO] Loading data for {subject_id}/{task}")
    print(f"[INFO] Mocap: {path_mocap}")
    print(f"[INFO] IK:    {path_ik}")

    # Load data (skip first 7 columns: freeflyer position + quaternion if start_dof=7)
    df_ik = pd.read_csv(path_ik).iloc[:, 7:]
    df_mocap = pd.read_csv(path_mocap).iloc[:, 7:]

    # Ensure same length
    min_len = min(df_ik.shape[0], df_mocap.shape[0])
    if df_ik.shape[0] != df_mocap.shape[0]:
        print(
            f"[WARNING] Length mismatch: IK={df_ik.shape[0]}, Mocap={df_mocap.shape[0]}. Truncating to {min_len}"
        )
        df_ik = df_ik.iloc[:min_len, :]
        df_mocap = df_mocap.iloc[:min_len, :]

    # Temporal synchronization using specified joint
    print(f"[INFO] Synchronizing signals using joint: {sync_joint}")
    if sync_joint not in df_ik.columns or sync_joint not in df_mocap.columns:
        print(
            f"[WARNING] Sync joint '{sync_joint}' not found. Skipping synchronization."
        )
        lag = 0
    else:
        knee_ik = df_ik[sync_joint].to_numpy()
        knee_mocap = df_mocap[sync_joint].to_numpy()

        lag = synchronize_signals(knee_ik, knee_mocap)

    print(f"[INFO] Temporal lag: {lag} frames")

    # Load joint trajectories
    q_ik = read_specific_joint(
        str(path_ik),
        DOFS,
        start_sample,
    )

    q_mocap = read_specific_joint(
        str(path_mocap),
        DOFS,
        start_sample,
    )

    # Apply temporal synchronization
    if lag > 0:
        # IK is delayed relative to mocap
        q_ik = q_ik[lag:, :]

    elif lag < 0:
        # Mocap is delayed relative to IK
        q_mocap = q_mocap[-lag:, :]

    # Ensure the same length after synchronization
    min_len = min(q_ik.shape[0], q_mocap.shape[0])

    q_ik = q_ik[:min_len, :]
    q_mocap = q_mocap[:min_len, :]

    # Filter joints to compare
    joint_indices = [
        i for i in range(start_dof, len(DOFS)) if DOFS[i] not in EXCLUDED_JOINTS
    ]
    joint_names = [DOFS[i] for i in joint_indices]

    print(f"[INFO] Comparing {len(joint_indices)} joints")

    # Compute metrics
    rmse_list = []
    mae_list = []
    corr_list = []
    unit_list = []

    for i in joint_indices:
        name = DOFS[i]

        error = q_mocap[:, i] - q_ik[:, i]

        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        # Convert only angular DOFs from radians to degrees
        if name.endswith("[rad]"):
            rmse = np.rad2deg(rmse)
            mae = np.rad2deg(mae)
            unit = "°"

        elif name.endswith("[m]"):
            unit = "m"

        else:
            # Quaternion components are dimensionless
            unit = ""

        corr_coef = np.corrcoef(
            q_mocap[:, i],
            q_ik[:, i],
        )[0, 1]

        rmse_list.append(rmse)
        mae_list.append(mae)
        corr_list.append(corr_coef)
        unit_list.append(unit)

    # Plotting
    if show_plots or save_plots:
        plot_dir = None
        if save_plots:
            plot_dir = output_root / subject_id / task / output_plot_dir
            plot_dir.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Saving plots to {plot_dir}")

        # Time-series plots
        n_per_fig = plots_per_figure
        for j, i in enumerate(joint_indices):
            name = DOFS[i]
            rmse = rmse_list[j]
            mae = mae_list[j]
            corr_coef = corr_list[j]
            unit = unit_list[j]

            unit_suffix = f" {unit}" if unit else ""

            # Create new figure every n_per_fig plots
            if j % n_per_fig == 0:
                fig, axs = plt.subplots(n_per_fig, 1, figsize=(10, 14))
                fig.tight_layout(pad=4.0)
                fig_num = j // n_per_fig

            ax = axs[j % n_per_fig] if n_per_fig > 1 else axs
            ax.plot(q_ik[:, i], label="IK", linewidth=2, color="green")
            ax.plot(q_mocap[:, i], label="Mocap (GT)", linewidth=2, color="red")
            ax.set_title(
                f"{name}\n"
                f"RMSE: {rmse:.2f}{unit_suffix}, "
                f"MAE: {mae:.2f}{unit_suffix}, "
                f"Corr: {corr_coef:.3f}",
                fontsize=10,
            )
            ax.set_xlabel("Frame")
            ax.set_ylabel("ylabel")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

            # Save or show figure after every n_per_fig plots or at the end
            if (j % n_per_fig == n_per_fig - 1) or (j == len(joint_indices) - 1):
                if save_plots:
                    fig_path = plot_dir / f"timeseries_{fig_num:02d}.png"
                    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                    print(f"[SAVED] {fig_path}")
                if show_plots:
                    plt.show()
                else:
                    plt.close(fig)

        # Bar chart of RMSEs
        # rmse_array = np.array(rmse_list)
        # avg_rmse = np.mean(rmse_array)

        angular_mask = np.array([name.endswith("[rad]") for name in joint_names])

        angular_rmse = np.asarray(rmse_list)[angular_mask]
        angular_mae = np.asarray(mae_list)[angular_mask]
        angular_corr = np.asarray(corr_list)[angular_mask]

        avg_rmse = np.mean(angular_rmse)
        std_rmse = np.std(angular_rmse)
        avg_mae = np.mean(angular_mae)
        avg_corr = np.nanmean(angular_corr)

        print("\n" + "=" * 60)
        print(f"COMPARISON RESULTS - {subject_id}/{task}")
        print("=" * 60)
        print(f"Average joint RMSE:        {avg_rmse:.2f}° ± {std_rmse:.2f}°")
        print(f"Average joint MAE:         {avg_mae:.2f}°")
        print(f"Average joint correlation: {avg_corr:.3f}")
        print("=" * 60)

        angular_names = [name for name in joint_names if name.endswith("[rad]")]

        angular_rmse = np.array(
            [
                rmse
                for name, rmse in zip(joint_names, rmse_list)
                if name.endswith("[rad]")
            ]
        )

        avg_rmse = np.mean(angular_rmse)

        plt.figure(figsize=(14, 6))

        bars = plt.bar(
            range(len(angular_names)),
            angular_rmse,
            color="skyblue",
            edgecolor="black",
        )

        plt.xticks(
            range(len(angular_names)),
            angular_names,
            rotation=45,
            ha="right",
        )

        plt.axhline(
            avg_rmse,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Average RMSE: {avg_rmse:.2f}°",
        )

        plt.ylabel("RMSE (degrees)")

        # Add value annotations
        for idx, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.5,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

        plt.ylabel("RMSE (degrees)", fontsize=12)
        plt.title(
            f"Joint Angle RMSEs - {subject_id}/{task}", fontsize=14, fontweight="bold"
        )
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(fontsize=11)
        plt.tight_layout()

        if save_plots:
            bar_path = plot_dir / "rmse_bar_chart.png"
            plt.savefig(bar_path, dpi=150, bbox_inches="tight")
            print(f"[SAVED] {bar_path}")
        if show_plots:
            plt.show()
        else:
            plt.close()

    # Print summary statistics
    rmse_array = np.array(rmse_list)
    mae_array = np.array(mae_list)
    corr_array = np.array(corr_list)

    avg_rmse = np.mean(rmse_array)
    std_rmse = np.std(rmse_array)
    avg_mae = np.mean(mae_array)
    avg_corr = np.mean(corr_array)

    print("\n" + "=" * 60)
    print(f"COMPARISON RESULTS - {subject_id}/{task}")
    print("=" * 60)
    print(f"Average RMSE:        {avg_rmse:.2f}° ± {std_rmse:.2f}°")
    print(f"Average MAE:         {avg_mae:.2f}°")
    print(f"Average Correlation: {avg_corr:.3f}")
    print("=" * 60 + "\n")

    # Per-joint detailed results
    print("Per-joint metrics:")
    print(f"{'DOF':<50} {'RMSE':>10} {'MAE':>10} {'Unit':>8} {'Corr':>8}")
    print("-" * 90)

    for name, rmse, mae, unit, corr in zip(
        joint_names, rmse_list, mae_list, unit_list, corr_list
    ):
        print(f"{name:<50} {rmse:>10.3f} {mae:>10.3f} {unit:>8} {corr:>8.3f}")

    print(f"[SUCCESS] {subject_id}/{task} comparison complete")
    return True


def main():
    args = parse_args()
    validate_inputs(args.subject_id, args.task)

    comfi_root = Path(args.comfi_root).resolve()
    output_root = Path(args.output_root).resolve()

    try:
        comparison_joint_angles(
            subject_id=args.subject_id,
            task=args.task,
            comfi_root=comfi_root,
            output_root=output_root,
            mocap_dir=args.mocap_dir,
            ik_angles_file=args.ik_angles_file,
            mocap_angles_file=args.mocap_angles_file,
            start_sample=args.start_sample,
            start_dof=args.start_dof,
            sync_joint=args.sync_joint,
            plots_per_figure=args.plots_per_figure,
            show_plots=args.show_plots,
            save_plots=args.save_plots,
            output_plot_dir=args.output_plot_dir,
        )
        print("[DONE] Comparison completed successfully")
    except Exception as e:
        print(f"[ERROR] Comparison failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
