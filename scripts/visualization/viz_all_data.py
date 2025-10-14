#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

from dataclasses import dataclass
from typing import Dict, List, Tuple
from xmlrpc.client import Boolean
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat
import meshcat_shapes

from comfi_examples.urdf_utils import build_human_model, load_robot_panda
from comfi_examples.viz_utils import box_between_frames, set_tf, draw_table, addViewerBox, make_visuals_gray, animate
from comfi_examples.utils import compute_time_sync,load_cameras_from_soder, load_robot_base_pose, load_all_data,load_force_data

SUBJECT_IDS = [
    "1012","1118","1508","1602","1847","2112","2198","2307","3361",
    "4162","4216","4279","4509","4612","4665","4687","4801","4827"
]
DS_TASKS = [
    "Screwing","ScrewingSat","Crouching","Picking","Hammering","HammeringSat","Jumping","Lifting",
    "QuickLifting","Lower","SideOverhead", "FrontOverhead","RobotPolishing","RobotWelding",
    "Polishing","PolishingSat","SitToStand","Squatting","Static","Upper","CircularWalking","StraightWalking",
    "Welding","WeldingSat"
]

TASKS_WITHOUT_FP = ["CircularWalking", "StraightWalking", "SideOverhead", "FrontOverhead"]

TASKS_WITH_TABLE = ["RobotPolishing","RobotWelding"]

URDF_DIR_DEFAULT   = Path("model/urdf")   # contains files like 4279_scaled.urdf
MESHES_DIR_DEFAULT = Path("model")        # root for visuals/collisions

def task_has_robot(task_name: str) -> bool:
    return "robot" in task_name.replace("-", "_").lower()

@dataclass(frozen=True)
class Paths:
    # Core args
    comfi_root: Path
    subject_id: str
    task: str        # normalized (e.g., "RobotWelding")
    freq: int        # 40 or 100
    freq_anim:str

    # files you consume
    mks_csv: Path
    q_ref_csv: Path
    cam0_ts_csv: Path
    jcp_mocap: Path
    soder_paths: Dict[str, Path]

    urdf_path: Path
    urdf_meshes_path: Path

    bool_table: Boolean

    # Resolved, OPTIONAL (present only for Robot* tasks or when files exist)
    force_data: Path | None
    robot_csv: Path | None
    robot_base_yaml: Path | None

    @property
    def split(self) -> str:
        return "aligned" if self.freq == 40 else "raw"

    @classmethod
    def from_args(
        cls,
        comfi_root: str | Path,
        subject_id: str,
        task: str,
        freq: int,
        urdf_dir: str | Path = URDF_DIR_DEFAULT,
        meshes_dir: str | Path = MESHES_DIR_DEFAULT,
        camera_ids=("0","2","4","6"),
                ) -> "Paths":

        root = Path(comfi_root).resolve()

        freq_anim = freq
        if int(freq)==100 and task_has_robot(task):
            print("[WARNING] 100Hz data for robot not available, fallback to 40Hz")
            freq_anim = 40
        split = "aligned" if (int(freq) == 40 or task_has_robot(task)) else "raw"

        # REQUIRED CSVs
        mks_csv = root / "mocap" / split / subject_id / task / "markers_trajectories.csv"
        if not mks_csv.exists():
            raise FileNotFoundError(f"Missing MKS mocap {split} CSV: {mks_csv}")

        q_ref_csv = root / "mocap" / split / subject_id / task / "joint_angles.csv"
        if not q_ref_csv.exists():
            raise FileNotFoundError(f"Missing joint angles mocap {split} CSV: {q_ref_csv}")

        cam0_ts_csv = root / "videos" / subject_id / task / "camera_0_timestamps.csv"
        if not cam0_ts_csv.exists():
            raise FileNotFoundError(f"Missing camera 0 timestamps CSV: {cam0_ts_csv}")

        jcp_mocap = root / "mocap" / split / subject_id / task / "joint_center_positions.csv"
        if not jcp_mocap.exists():
            raise FileNotFoundError(f"Missing JCP mocap {split} CSV: {jcp_mocap}")

        soder: Dict[str, Path] = {}
        for cid in camera_ids:
            cand = root / "cam_params" / subject_id / "extrinsics" / "cam_to_world" / f"camera_{cid}" / "soder.txt"
            if cand.exists():
                soder[cid] = cand.resolve()
            else:
                raise FileNotFoundError(f"Missing Soder camera {cid} file: {cand}")

        # REQUIRED URDF & meshes, derived from subject id
        urdf_dir = Path(urdf_dir).resolve()
        meshes_dir = Path(meshes_dir).resolve()
        urdf_path = urdf_dir / f"{subject_id}_scaled.urdf"
        if not urdf_path.exists():
            raise FileNotFoundError(
                f"Missing URDF for subject {subject_id}: {urdf_path}\n"
                f"(Edit URDF_DIR_DEFAULT / filename pattern if needed.)"
            )
        if not meshes_dir.exists():
            raise FileNotFoundError(
                f"Meshes directory not found: {meshes_dir}\n"
                f"(Edit MESHES_DIR_DEFAULT if your repo differs.)"
            )

        if task in TASKS_WITHOUT_FP:
            force_data = None
        else:
            force_data = root / "forces" / split / subject_id / task / f"{task}_devices.csv"
            if not force_data.exists():
                raise FileNotFoundError(f"Missing force plates CSV: {force_data}")

        # OPTIONAL robot assets (only for Robot* tasks)
        if task_has_robot(task):
            robot_csv = root / "robot" / split / subject_id / f"{subject_id}_{task}.csv"
            if not robot_csv.exists():
                raise FileNotFoundError(f"CSV not found for robot for task {task} and id {subject_id}: {robot_csv}")
            robot_base = root / "robot" / "robot_in_world" / subject_id / "robot_base_pose.yaml"
            if not robot_base.exists():
                raise FileNotFoundError(f"YAML not found for robot base pose for id {subject_id}: {robot_base}")
        else:
            robot_csv = None
            robot_base = None

        if task in TASKS_WITH_TABLE:
            bool_table = True
        else:
            bool_table = False

        return cls(
            comfi_root=root,
            subject_id=str(subject_id),
            task=task,
            freq=int(freq),
            freq_anim=int(freq_anim),
            mks_csv=mks_csv,
            q_ref_csv=q_ref_csv,
            cam0_ts_csv=cam0_ts_csv,
            jcp_mocap=jcp_mocap,
            soder_paths=soder,
            urdf_path=urdf_path,
            urdf_meshes_path=meshes_dir,
            bool_table=bool_table,
            force_data=force_data,
            robot_csv=robot_csv,
            robot_base_yaml=robot_base
        )

@dataclass
class Scene:
    viewer: meshcat.Visualizer
    viz_human: MeshcatVisualizer
    viz_robot: MeshcatVisualizer
    human_model: pin.Model
    human_data: pin.Data
    robot_model: pin.Model
    robot_data: pin.Data

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualize COMFI multimodal data (mocap, forces, robot) in Meshcat."
    )
    p.add_argument("--id", dest="subject_id", required=True,
                   help="ID (e.g., 1012)")
    p.add_argument("--task", required=True,
                   help="Task name (e.g., RobotWelding)")
    p.add_argument("--comfi-root", default=Path(os.environ.get("COMFI_ROOT", "COMFI")),
                   help="Path to COMFI dataset root.")
    p.add_argument("--freq", type=int, choices=[40, 100], required=True,
                   help="Sampling frequency: 40 (aligned) or 100 (raw).")
    p.add_argument("--start", type=int, default=0,
                   help="Start frame index (inclusive). Default: 0")
    p.add_argument("--stop", type=int, default=None,
                   help="Stop frame index (exclusive). Default: None (till end)")

    p.add_argument("--with-jcp-hpe", action="store_true",
                   help="Enable JCP HPE mode (default: False)")
    p.add_argument("--jcp-hpe-mode", choices=["aligned", "2cams", "4cams","3cams"], default=None,
                   help="Specify JCP HPE mode if --with-jcp-hpe is set.")

    args = p.parse_args()
    if args.with_jcp_hpe and args.jcp_hpe_mode is None:
        p.error("--jcp-hpe-mode is required when --with-jcp-hpe is set.")

    return p.parse_args()


def define_scene(urdf_path: str,
                 urdf_meshes_path: str,
                 bool_table: Boolean,
                 T_world_robot: np.ndarray,
                 cameras: Dict[str, np.ndarray],
                 forceplates_dims_and_centers: Tuple[List[Tuple[float,float]], List[Tuple[float,float,float]]],
                 bg_top=(1,1,1), bg_bottom=(1,1,1), grid_height=-0.0) -> Scene:
    """
    Builds:
      - Meshcat viewer (shared)
      - Human model (locked joints + gray)
      - Panda model (robot)
      - Background, grid, force plates, camera boxes + frames + links, labels, world frames
    Returns handles so the rest of the code is clean.
    """
    # Human base
    model_h, coll_h, vis_h, _ = build_human_model(urdf_path, urdf_meshes_path)
    data_h = model_h.createData()

    # make visuals gray
    make_visuals_gray(vis_h)

    # Lock joints
    # joints_to_lock = [
    #     "middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z",
    #     "left_wrist_X", "left_wrist_Z", "right_wrist_X", "right_wrist_Z"
    # ]
    # model_h, coll_h, vis_h, data_h = lock_joints(model_h, coll_h, vis_h, joints_to_lock)


    # Shared Meshcat
    viewer = meshcat.Visualizer()

    # Visualizers
    viz_human = MeshcatVisualizer(model_h, coll_h, vis_h)
    viz_human.initViewer(viewer, open=True)
    viz_human.viewer.delete()  # clear if relaunch
    viz_human.loadViewerModel("ref")
    viz_human.display(pin.neutral(model_h))


    if T_world_robot is not None:
        # Panda
        model_r, coll_r, vis_r, data_r = load_robot_panda()
        viz_robot = MeshcatVisualizer(model_r, coll_r, vis_r)
        viz_robot.initViewer(viewer)
        viz_robot.loadViewerModel(rootNodeName="panda")
        viz_robot.viewer["panda"].set_transform(T_world_robot)
    else:
        viz_robot = None
        model_r = None
        data_r = None

    # Background/grid
    native_viz = viz_human.viewer
    native_viz["/Background"].set_property("top_color", list(bg_top))
    native_viz["/Background"].set_property("bottom_color", list(bg_bottom))
    native_viz["/Grid"].set_transform(np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, grid_height],
        [0, 0, 0, 1]
    ]))

    # Force plates
    R_frame = np.array([[0, -1, 0],
                    [-1,  0, 0],
                    [0,   0, -1]])
    fp_dim, fp_centers = forceplates_dims_and_centers
    for j, ((sx, sy), (cx, cy, cz)) in enumerate(zip(fp_dim, fp_centers), start=1):
        name = f"fp{j}"
        frame = f'R_fp{j}'
        text_name  = f"fp{j}_text"
        addViewerBox(viz_human, name, sx, sy, 0.01, rgba=[0.5, 0.5, 0.5, 1.0])
        T = np.eye(4)
        T[:3, 3] = [cx, cy, cz + 0.01/2.0]
        set_tf(viz_human, name, T)

        meshcat_shapes.frame(
        viz_human.viewer[frame],
        axis_length=0.2,
        axis_thickness=0.02,
        opacity=0.8,
        origin_radius=0.02
        )
        T_frame = np.eye(4)
        T_frame[:3, :3] = R_frame
        T_frame[:3, 3] = [cx, cy, cz + 0.03 ]
        set_tf(viz_human, frame, T_frame)

        # 3) Label texte "fpj" décalé en XY et légèrement au-dessus
        meshcat_shapes.textarea(viz_human.viewer[text_name], f"fp{j}", font_size=32)
        T_text = np.eye(4)
        T_text[:3, 3] = [cx + 0.05, cy + 0.05, cz + 0.03]
        set_tf(viz_human, text_name, T_text)

    # Some frames and labels
    meshcat_shapes.textarea(viz_human.viewer["R_world_text"], "R0", font_size=32)
    T_name = np.eye(4)
    T_name[0,3]+=0.15
    T_name[1,3]+=0.15
    T_name[2,3]+=0.15
    set_tf(viz_human,  "R_world_text", T_name)
    meshcat_shapes.frame(viz_human.viewer["R_world"], axis_length=0.4, axis_thickness=0.04, opacity=1, origin_radius=0.02)

    if T_world_robot is not None:
        meshcat_shapes.frame(viz_human.viewer["R_robot"], axis_length=0.4, axis_thickness=0.02, opacity=1, origin_radius=0.02)
        set_tf(viz_human, "R_robot", T_world_robot)
        meshcat_shapes.textarea(viz_human.viewer["Rrobot_text"], "Rrobot", font_size=28)
        T_name = np.eye(4)
        T_name[0:3,3]=T_world_robot[0:3,3]
        T_name[0,3]+=0.25
        T_name[1,3]-=0.25
        T_name[2,3]+=0.025
        set_tf(viz_human,  "Rrobot_text", T_name)

    # Camera boxes + frames + links + text
    # Expect keys like "c0","c2","c4","c6"
    cam_order = sorted(cameras.keys())  # stable order
    def cam_frame_name(k): return f"f_cam_{k}"
    def cam_text_name(k):  return f"R_c{k}_text"
    def cam_box_name(k):   return f"cam_{k}"

    # optional: connect pairs with links if c0<->c2 and c4<->c6 exist
    def safe_box_between(a, b, name):
        if a in cameras and b in cameras:
            box_between_frames(viz_human, name, cameras[a], cameras[b], thickness=0.1, height=0.1, rgba=(0.01,0.01,0.01,0.9))

    safe_box_between("0","2","link_c0_c2")
    safe_box_between("4","6","link_c4_c6")

    for k in cam_order:
        T_cam=np.eye(4)
        T_cam[0:3,0:3] =[[1,0,0],
                            [0,0,-1],
                            [0,1,0]]

        Tck = cameras[k]
        meshcat_shapes.frame(viz_human.viewer[cam_frame_name(k)], axis_length=0.2, axis_thickness=0.02, opacity=0.8, origin_radius=0.02)
        set_tf(viz_human, cam_frame_name(k), Tck)

        addViewerBox(viz_human, cam_box_name(k), 0.1, 0.1, 0.1, rgba=[0.01, 0.01, 0.01, 1.0])
        set_tf(viz_human, cam_box_name(k), Tck)

        meshcat_shapes.textarea(viz_human.viewer[cam_text_name(k)], f"cam{k}", font_size=28)
        Ttxt = np.array(Tck)
        Ttxt = Ttxt.copy()
        T_cam[0:3,3]=Ttxt[0:3,3]
        T_cam[2, 3] += 0.1
        set_tf(viz_human, cam_text_name(k), T_cam)

    # Table
    if bool_table:
        T_world_table = np.eye(4)
        T_world_table[:3, 3] = [0.9, -0.6, 0.0]
        draw_table(viz_human, T_world_table)

    return Scene(
        viewer=viewer,
        viz_human=viz_human,
        viz_robot=viz_robot,
        human_model=model_h,
        human_data=data_h,
        robot_model=model_r,
        robot_data=data_r
    )

def main():
    args = parse_args()

    # Minimal friendly validation
    if args.subject_id not in SUBJECT_IDS:
        raise ValueError(f"Unknown subject ID '{args.subject_id}'. Allowed: {', '.join(SUBJECT_IDS)}")
    if args.task not in DS_TASKS:
        raise ValueError(f"Unknown task '{args.task}'. "
                         f"Allowed: {', '.join(DS_TASKS)}")

    paths = Paths.from_args(args.comfi_root, args.subject_id, args.task, args.freq)
    if args.with_jcp_hpe:
        Paths.jcp_hpe = Path("output").resolve() / "res_hpe" / args.subject_id / args.task  / f"3d_keypoints_{args.jcp_hpe_mode}.csv"

    #read all data
    payload = load_all_data(paths, start_sample=0, converter=1000.0)
    mks_dict = payload["mks_dict"]
    mks_names = payload["mks_names"]
    q_ref = payload["q_ref"]
    q_robot = payload["q_robot"]
    t_cam = payload["t_cam"]
    t_robot = payload["t_robot"]
    jcp_mocap = payload["jcp_mocap"]
    jcp_names = payload["jcp_names"]
    jcp_hpe = None
    jcp_names_hpe = None

    if args.with_jcp_hpe:
        jcp_hpe = payload.get("jcp_hpe", None)
        jcp_names_hpe = payload.get("jcp_names_hpe", None)


    # transforms (robot base + cameras)
    if paths.robot_base_yaml is not None:
        T_world_robot = load_robot_base_pose(paths.robot_base_yaml)
    else :
        T_world_robot = None

    cameras = load_cameras_from_soder(paths.soder_paths)

    if paths.force_data is not None:
        force_data = load_force_data(paths.force_data)
    else :
        force_data = None


    # define the scene
    fp_dims = [(0.5,0.6), (0.50,0.60), (0.50,0.60), (0.9,1.8), (0.5,0.6)]
    fp_centers = [(-0.830,-0.3,0.0), (-0.25,-0.3,0.0), (0.39,-0.3,0.0), (-1.68,-0.3,0.0), (-0.25,0.3,0.0)]

    scene = define_scene(
        urdf_path=paths.urdf_path,
        urdf_meshes_path=paths.urdf_meshes_path,
        bool_table=paths.bool_table,
        T_world_robot=T_world_robot,
        cameras=cameras,
        forceplates_dims_and_centers=(fp_dims, fp_centers),
        bg_top=(1,1,1), bg_bottom=(1,1,1), grid_height=-0.0
    )

    #time syn between cameras and robot data
    if t_robot is not None:
        sync = compute_time_sync(t_cam, t_robot, tol_ms=5)
        if sync:
            print("Synced at:", sync)
        else:
            print("No time sync match found (even within tolerance).")
            sync = None
    else:
        sync = None

    #animation
    animate(scene, jcp_mocap, jcp_names,jcp_hpe,jcp_names_hpe, q_ref, q_robot, force_data,
        (fp_dims, fp_centers), sync, paths.freq_anim, step=5, i0=0)

if __name__ == "__main__":
    main()
