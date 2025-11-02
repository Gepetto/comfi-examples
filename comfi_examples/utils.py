import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


def to_utc(s: pd.Series) -> pd.Series:
    return s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")


def load_transformation(file_path):
    """
    Loads the transformation parameters (R, d, s, rms) from a text file.

    Parameters:
    file_path: str
        Path to the file from which the transformation parameters will be read.

    Returns:
    R: ndarray
        Rotation matrix (3x3)
    d: ndarray
        Translation vector (3,)
    s: float
        Scale factor
    rms: float
        Root mean square fit error
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
        R_start = lines.index("Rotation Matrix (R):\n") + 1
        R = np.loadtxt(lines[R_start : R_start + 3])
        d_start = lines.index("Translation Vector (d):\n") + 1
        d = np.loadtxt(lines[d_start : d_start + 1]).flatten()
        s_line = next(line for line in lines if line.startswith("Scale Factor (s):"))
        s = float(s_line.split(":")[1].strip())
        rms_line = next(line for line in lines if line.startswith("RMS Error:"))
        rms = float(rms_line.split(":")[1].strip())
    return R, d, s, rms


def udp_csv_to_dataframe(csv_path, marker_names):
    """
    Preprocess a UDP CSV file into a DataFrame suitable for read_mks_data.

    Parameters:
        csv_path (str): Path to the CSV file.
        marker_names (list): List of marker base names (without _x/_y/_z).

    Returns:
        pd.DataFrame: A DataFrame with columns formatted as marker_x, marker_y, marker_z.
    """
    # 1. Open manually
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # 2. Skip the header
    lines = lines[:]

    # 3. Prepare all rows
    all_rows = []
    for line in lines:
        # Remove newline, then split
        line = line.strip()
        if not line:
            continue  # skip empty lines
        parts = line.split(",")
        udp_values = [float(val) for val in parts[2:]]
        all_rows.append(udp_values)

    # 4. Now create a dataframe
    udp_df = pd.DataFrame(all_rows)

    # 5. Build column names
    new_columns = []
    for marker in marker_names:
        new_columns.extend([f"{marker}_x", f"{marker}_y", f"{marker}_z"])

    if udp_df.shape[1] != len(new_columns):
        raise ValueError(
            f"Mismatch between expected markers ({len(new_columns)}) and data columns ({udp_df.shape[1]}). Check marker list!"
        )

    udp_df.columns = new_columns

    return udp_df


def read_mks_data(data_markers, start_sample=0, converter=1.0):
    # the mks are ordered in a csv like this : "time,r.ASIS_study_x,r.ASIS_study_y,r.ASIS_study_z...."
    """
    Parameters:
        data_markers (pd.DataFrame): The input DataFrame containing marker data.
        start_sample (int): The index of the sample to start processing from.
        time_column (str): The name of the time column in the DataFrame.

    Returns:
        list: A list of dictionaries where each dictionary contains markers with 3D coordinates.
        dict: A dictionary representing the markers and their 3D coordinates for the specified start_sample.
    """
    # Extract marker column names
    marker_columns = [
        col[:-6] for col in data_markers.columns if col.endswith("_X[mm]")
    ]

    # Initialize the result list
    result_markers = []

    # Iterate over each row in the DataFrame
    for _, row in data_markers.iterrows():
        frame_dict = {}
        for marker in marker_columns:
            x = row[f"{marker}_X[mm]"] / converter  # convert to m
            y = row[f"{marker}_Y[mm]"] / converter
            z = row[f"{marker}_Z[mm]"] / converter
            frame_dict[marker] = np.array([x, y, z])  # Store as a NumPy array
        result_markers.append(frame_dict)

    # Get the data for the specified start_sample
    start_sample_mks = result_markers[start_sample]

    return result_markers, start_sample_mks


def try_read_mks(data_or_path, **kwargs):
    """
    Funnel ALL reads through read_mks_data.
    - If given a path, load CSV -> DataFrame, then pass to read_mks_data.
    - If read_mks_data doesn't apply (e.g., it's a plain table), return the DataFrame.
    """
    if isinstance(data_or_path, (str, os.PathLike)):
        df = pd.read_csv(
            data_or_path, **{k: v for k, v in kwargs.items() if k in {"parse_dates"}}
        )
    else:
        df = data_or_path

    try:
        # Try using your canonical reader first
        return read_mks_data(
            df, **{k: v for k, v in kwargs.items() if k != "parse_dates"}
        )
    except Exception:
        # Fall back to raw DF if this CSV isn't an MKS blob
        return df


def load_cameras_from_soder(soder_paths):
    cams = {}
    for key, path in soder_paths.items():
        R, d, _, _ = load_transformation(path)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = d.reshape(3)
        cams[key] = T
    return cams


def load_robot_base_pose(yaml_path: str) -> np.ndarray:
    with open(yaml_path) as f:
        Y = yaml.safe_load(f)["world_T_robot"]
    R = np.array(Y["rotation_matrix"], dtype=float)
    t = np.array(Y["translation"], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def load_all_data(paths, start_sample: int = 0, converter: float = 1000.0):
    # mks mocap + names
    mks_raw = pd.read_csv(paths.mks_csv)  # still funnel via try_read_mks next line
    mks_dict, start_sample_dict = try_read_mks(
        mks_raw, start_sample=start_sample, converter=converter
    )
    mks_names = list(start_sample_dict.keys())

    # Reference human joint
    q_ref_df = try_read_mks(paths.q_ref_csv)
    q_ref = (
        q_ref_df
        if isinstance(q_ref_df, np.ndarray)
        else pd.read_csv(paths.q_ref_csv).to_numpy(dtype=float)
    )

    # Robot CSV
    if paths.robot_csv is not None:
        robot_df = try_read_mks(paths.robot_csv, parse_dates=["_cam_time", "timestamp"])
        if not isinstance(robot_df, pd.DataFrame):
            robot_df = pd.read_csv(
                paths.robot_csv, parse_dates=["_cam_time", "timestamp"]
            )
        pos_cols = [f"panda_joint{i}_position[rad]" for i in range(1, 8)]
        q_robot = robot_df[pos_cols].to_numpy(dtype=float)
        q_robot = np.hstack(
            [q_robot, np.zeros((q_robot.shape[0], 2), dtype=q_robot.dtype)]
        )
    else:
        robot_df = None
        q_robot = None

    # Camera timestamps + robot timestamps (time sync)
    t_cam = try_read_mks(paths.cam0_ts_csv, parse_dates=["timestamp"])
    if not isinstance(t_cam, pd.DataFrame):
        t_cam = pd.read_csv(paths.cam0_ts_csv, parse_dates=["timestamp"])

    if paths.robot_csv is not None:
        t_robot = try_read_mks(paths.robot_csv, parse_dates=["timestamp"])
        if not isinstance(t_robot, pd.DataFrame):
            t_robot = pd.read_csv(paths.robot_csv, parse_dates=["timestamp"])
    else:
        t_robot = None

    # Joint Center Positions (JCP) from mocap
    jcp_raw = pd.read_csv(paths.jcp_mocap)
    jcp_dict, start_sample_jcp_dict = try_read_mks(
        jcp_raw, start_sample=start_sample, converter=converter
    )
    jcp_names = list(start_sample_jcp_dict.keys())

    # JCP from hpe
    jcp_dict_hpe = None
    jcp_names_hpe = None
    if (
        hasattr(paths, "jcp_hpe")
        and paths.jcp_hpe is not None
        and paths.jcp_hpe.exists()
    ):
        jcp_raw_hpe = pd.read_csv(paths.jcp_hpe)
        jcp_dict_hpe, start_sample_jcp_dict_hpe = read_mks_data(
            jcp_raw_hpe, start_sample=start_sample, converter=converter
        )
        _jcp_hpe = [
            start_sample_jcp_dict_hpe[name] for name in start_sample_jcp_dict_hpe.keys()
        ]
        jcp_names_hpe = [name + "_hpe" for name in start_sample_jcp_dict_hpe.keys()]

    return {
        "mks_dict": mks_dict,
        "mks_names": mks_names,
        "q_ref": q_ref,
        "robot_df": robot_df,
        "q_robot": q_robot,
        "t_cam": t_cam,
        "t_robot": t_robot,
        "jcp_mocap": jcp_dict,
        "jcp_names": jcp_names,
        "jcp_hpe": jcp_dict_hpe,
        "jcp_names_hpe": jcp_names_hpe,
    }

def load_all_data_comparison(paths, start_sample: int = 0, converter: float = 1000.0):
    # mks mocap + names
    mks_raw = pd.read_csv(paths.mks_csv)  # still funnel via try_read_mks next line
    mks_dict, start_sample_dict = try_read_mks(
        mks_raw, start_sample=start_sample, converter=converter
    )
    mks_names = list(start_sample_dict.keys())

    # Reference human joint
    q_ref_df = try_read_mks(paths.q_ref_csv)
    q_ref = (
        q_ref_df
        if isinstance(q_ref_df, np.ndarray)
        else pd.read_csv(paths.q_ref_csv).to_numpy(dtype=float)
    )

    # comparative human joint
    q_swika_df = try_read_mks(paths.q_swika_csv)
    q_swika = (
        q_swika_df
        if isinstance(q_swika_df, np.ndarray)
        else pd.read_csv(paths.q_swika_csv).to_numpy(dtype=float)
    )

    # Robot CSV
    if paths.robot_csv is not None:
        robot_df = try_read_mks(paths.robot_csv, parse_dates=["_cam_time", "timestamp"])
        if not isinstance(robot_df, pd.DataFrame):
            robot_df = pd.read_csv(
                paths.robot_csv, parse_dates=["_cam_time", "timestamp"]
            )
        pos_cols = [f"panda_joint{i}_position[rad]" for i in range(1, 8)]
        q_robot = robot_df[pos_cols].to_numpy(dtype=float)
        q_robot = np.hstack(
            [q_robot, np.zeros((q_robot.shape[0], 2), dtype=q_robot.dtype)]
        )
    else:
        robot_df = None
        q_robot = None

    # Camera timestamps + robot timestamps (time sync)
    t_cam = try_read_mks(paths.cam0_ts_csv, parse_dates=["timestamp"])
    if not isinstance(t_cam, pd.DataFrame):
        t_cam = pd.read_csv(paths.cam0_ts_csv, parse_dates=["timestamp"])

    if paths.robot_csv is not None:
        t_robot = try_read_mks(paths.robot_csv, parse_dates=["timestamp"])
        if not isinstance(t_robot, pd.DataFrame):
            t_robot = pd.read_csv(paths.robot_csv, parse_dates=["timestamp"])
    else:
        t_robot = None

    # Joint Center Positions (JCP) from mocap
    jcp_raw = pd.read_csv(paths.jcp_mocap)
    jcp_dict, start_sample_jcp_dict = try_read_mks(
        jcp_raw, start_sample=start_sample, converter=converter
    )
    jcp_names = list(start_sample_jcp_dict.keys())

    # JCP from hpe
    jcp_dict_hpe = None
    jcp_names_hpe = None
    if (
        hasattr(paths, "jcp_hpe")
        and paths.jcp_hpe is not None
        and paths.jcp_hpe.exists()
    ):
        jcp_raw_hpe = pd.read_csv(paths.jcp_hpe)
        jcp_dict_hpe, start_sample_jcp_dict_hpe = read_mks_data(
            jcp_raw_hpe, start_sample=start_sample, converter=converter
        )
        _jcp_hpe = [
            start_sample_jcp_dict_hpe[name] for name in start_sample_jcp_dict_hpe.keys()
        ]
        jcp_names_hpe = [name + "_hpe" for name in start_sample_jcp_dict_hpe.keys()]

    return {
        "mks_dict": mks_dict,
        "mks_names": mks_names,
        "q_ref": q_ref,
        "q_swika": q_swika,
        "robot_df": robot_df,
        "q_robot": q_robot,
        "t_cam": t_cam,
        "t_robot": t_robot,
        "jcp_mocap": jcp_dict,
        "jcp_names": jcp_names,
        "jcp_hpe": jcp_dict_hpe,
        "jcp_names_hpe": jcp_names_hpe,
    }

def compute_time_sync(t_cam: pd.DataFrame, t_robot: pd.DataFrame, tol_ms: int = 5):
    t_cam = t_cam.copy()
    t_robot = t_robot.copy()
    t_cam["timestamp"] = to_utc(t_cam["timestamp"])
    t_robot["timestamp"] = to_utc(t_robot["timestamp"])
    t_cam = t_cam.reset_index().rename(columns={"index": "cam_idx"})
    t_robot = t_robot.reset_index().rename(columns={"index": "robot_idx"})

    exact = t_cam.merge(t_robot, on="timestamp", how="inner")
    if not exact.empty:
        first = exact.sort_values("timestamp").iloc[0]
        return {"cam_idx": int(first["cam_idx"]), "robot_idx": int(first["robot_idx"])}

    tol = pd.Timedelta(f"{tol_ms}ms")
    nearest = pd.merge_asof(
        t_cam.sort_values("timestamp"),
        t_robot.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=tol,
        suffixes=("_cam", "_robot"),
    ).dropna(subset=["robot_idx"])
    if not nearest.empty:
        first = nearest.iloc[0]
        return {"cam_idx": int(first["cam_idx"]), "robot_idx": int(first["robot_idx"])}
    return None


def read_mmpose_file(nom_fichier):
    donnees = []
    with open(nom_fichier, "r") as f:
        for ligne in f:
            ligne = ligne.strip().split(",")
            donnees.append([float(valeur) for valeur in ligne[1:]])
        # print('donnees=',donnees)
    return donnees


def read_mmpose_scores(liste_fichiers):
    all_scores = []
    for f in liste_fichiers:
        data = np.loadtxt(f, delimiter=",")
        all_scores.append(data[:, 0])
    return np.array(all_scores).transpose().tolist()


def save_to_csv(data, output_path, header=None):
    """Save 3D keypoints to a CSV file with optional header."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, header=header if header is not None else False)
    print(f"Saved {len(data)} frames to {output_path}")


def transform_keypoints_list_cam0_to_mocap(keypoints_list, R_trans, d_trans):
    """Apply transformation to each frame of flattened 3D keypoints."""
    transformed_list = []

    for flat_coords in keypoints_list:
        # Convert to shape (N, 3)
        p3d_cam0 = np.array(flat_coords).reshape(-1, 3)  # (N, 3)
        # Apply transformation
        p3d_mocap = (
            (R_trans @ p3d_cam0.T).T + d_trans
        ) * 1000  # conversion to mm,  (N, 3)
        # Flatten again
        transformed_list.append(p3d_mocap.flatten().tolist())

    return transformed_list


def load_force_data(csv_file_path, max_pf=5):
    df = pd.read_csv(csv_file_path)
    force_data = {}

    for sensor_id in range(1, max_pf + 1):
        sensor_name = f"Sensix_{sensor_id}"

        required_cols = [
            f"{sensor_name}_{axis}"
            for axis in ["Fx[N]", "Fy[N]", "Fz[N]", "Mx[N.mm]", "My[N.mm]", "Mz[N.mm]"]
        ]
        if all(col in df.columns for col in required_cols):
            force_data[sensor_id] = {
                "frames": df["camera_frame"].values
                if "camera_frame" in df.columns
                else None,
                "Fx": df[f"{sensor_name}_Fx[N]"].values,
                "Fy": df[f"{sensor_name}_Fy[N]"].values,
                "Fz": df[f"{sensor_name}_Fz[N]"].values,
                "Mx": df[f"{sensor_name}_Mx[N.mm]"].values,
                "My": df[f"{sensor_name}_My[N.mm]"].values,
                "Mz": df[f"{sensor_name}_Mz[N.mm]"].values,
            }

    return force_data


def kabsch_global(P_cam_seq, P_mocap_seq, weights=None):
    """
    P_cam_seq, P_mocap_seq: arrays (T, N, 3) temporally and point-wise aligned.
    Computes a single (R, t) that aligns everything (cam -> mocap) by minimizing the sum of errors.

    """

    assert P_cam_seq.shape == P_mocap_seq.shape and P_cam_seq.shape[-1] == 3
    T, N, _ = P_cam_seq.shape
    X = P_cam_seq.reshape(T * N, 3)
    Y = P_mocap_seq.reshape(T * N, 3)

    if weights is not None:
        w = np.asarray(weights).reshape(T, N)
        w = w / (w.sum() + 1e-12)
        w = w.reshape(T * N, 1)
        Xc = (X * w).sum(axis=0)  # weighted means
        Yc = (Y * w).sum(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = (Y0 * w).T @ X0
    else:
        Xc = X.mean(axis=0)
        Yc = Y.mean(axis=0)
        X0 = X - Xc
        Y0 = Y - Yc
        H = Y0.T @ X0

    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    t = Yc - R @ Xc

    X_align = (R @ X.T).T + t
    rms = np.sqrt(np.mean(np.sum((X_align - Y) ** 2, axis=1)))
    return R, t, rms


def plot_aligned_markers(P_mocap_seq, P_hpe_seq_aligned, marker_names):
    if isinstance(P_hpe_seq_aligned, pd.DataFrame):
        P_hpe_seq_aligned = P_hpe_seq_aligned.to_numpy().reshape(P_mocap_seq.shape)

    T, N, _ = P_mocap_seq.shape
    fig, axes = plt.subplots(N, 3, figsize=(12, 3 * N), sharex=True)
    if N == 1:  # cas spécial 1 marker
        axes = axes.reshape(1, 3)

    for i, mk in enumerate(marker_names):
        for j, coord in enumerate(["x", "y", "z"]):
            ax = axes[i, j]
            ax.plot(
                P_mocap_seq[:, i, j], "r", label="mocap" if i == 0 and j == 0 else ""
            )
            ax.plot(
                P_hpe_seq_aligned[:, i, j],
                "g",
                label="hpe" if i == 0 and j == 0 else "",
            )
            ax.set_title(f"{mk} - {coord}")
            if i == N - 1:
                ax.set_xlabel("Frame")
    axes[0, 0].legend()
    plt.tight_layout()
    plt.show()


def compute_mpjpe(P_pred, P_gt):
    """
    P_pred, P_gt: arrays (T, N, 3)
    Returns mean per-joint position error
    """
    assert P_pred.shape == P_gt.shape
    errors = np.linalg.norm(P_pred - P_gt, axis=2)  # (T,N) distance per joint per frame
    mpjpe = errors.mean()
    return mpjpe
