#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2
import numpy as np
from rtmlib import draw_skeleton, Custom, PoseTracker
from functools import partial
import csv
import os

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

DEFAULT_CAM_IDS = [0, 2, 4, 6]

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the RTMlib pose estimator on the set of videos of the dataset and create corresponding files for it"
    )
    # Allow multiple IDs / tasks (space-separated)
    p.add_argument("--id", dest="subject_ids", nargs="+", required=True,
                   help="Subject IDs (space-separated), e.g., --id 1012 1118")
    p.add_argument("--task", dest="tasks", nargs="+", required=True,
                   help="Task names (space-separated), e.g., --task RobotWelding Lifting")
    p.add_argument("--comfi-root", required=True,
                   help="Path to COMFI dataset root.")
    p.add_argument("--show_realtime", action='store_false', default=True,
                   help="Show the result in realtime. Default: True")
    # NEW: choose camera ids; default is all four
    p.add_argument("--cams", dest="cam_ids", nargs="+", type=int, choices=DEFAULT_CAM_IDS,
                   default=DEFAULT_CAM_IDS,
                   help="Camera IDs to process (space-separated). Default: 0 2 4 6")
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

def process_one(comfi_root, sid, task, show_realtime, cam_ids):
    # ---------------- Config ---------------- #
    device = 'cpu'  # 'cpu', 'cuda'
    backend = 'onnxruntime'
    openpose_skeleton = False

    # ----------------rtmlib-------------------- #
    # refer to rtmlib repo for more details
    custom = partial(
        Custom,
        to_openpose=openpose_skeleton,
        det_class='YOLOX',
        det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
        det_input_size=(640, 640),
        pose_class='RTMPose',
        pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip',
        pose_input_size=(192, 256),
        backend=backend,
        device=device
    )

    pose_tracker = PoseTracker(
        custom,
        det_frequency=10,
        tracking=False,
        tracking_thr=0.1,
        to_openpose=openpose_skeleton,
        backend=backend,
        device=device
    )

    for cam_id in cam_ids:
        print(f"\n=== Processing camera {cam_id} for id {sid} and task {task} ===")

        video_path = comfi_root / "videos" / sid / task / f'camera_{cam_id}.mp4'

        out_dir = Path(f"output/videos/{sid}/{task}").resolve()
        os.makedirs(out_dir, exist_ok=True)
        output_path = out_dir / f'camera_{cam_id}_with_keypoints.avi'

        csv_dir = Path(f"output/res_hpe/{sid}/{task}").resolve()
        os.makedirs(csv_dir, exist_ok=True)
        csv_output = csv_dir / f'keypoints_cam{cam_id}.csv'

        if not os.path.exists(video_path):
            print(f"⚠️  Video not found: {video_path}")
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_id = 0
        person_index = None  # keep the first detected person

        with open(csv_output, mode='w', newline='') as f:
            writer = csv.writer(f)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                keypoints, scores = pose_tracker(frame)

                if keypoints.shape[0] > 0:
                    if frame_id == 0:
                        person_index = 0  # choose the first person

                    if person_index is not None and person_index < keypoints.shape[0]:
                        keypoints = keypoints[person_index:person_index+1]
                        scores = scores[person_index:person_index+1]
                    else:
                        keypoints, scores = None, None

                    if keypoints is not None:
                        frame_with_skeleton = draw_skeleton(
                            frame.copy(), keypoints, scores, kpt_thr=0.5
                        )
                    else:
                        frame_with_skeleton = frame.copy()
                else:
                    frame_with_skeleton = frame.copy()

                out.write(frame_with_skeleton)
                print(f"[cam {cam_id}] Frame #{frame_id}")
                frame_id += 1

                if keypoints is not None and scores is not None:
                    keypoints_flat = keypoints.flatten().tolist()
                    scores_flat = scores.flatten().tolist()
                    scores_mean = [np.mean(scores_flat)]
                    writer.writerow(scores_mean + keypoints_flat)

                # Optional real-time display
                if show_realtime:
                    cv2.imshow(f'Skeleton Video - cam {cam_id}', frame_with_skeleton)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        out.release()

    if show_realtime:
        cv2.destroyAllWindows()

    print(f"\n✅ Processing finished for cameras {cam_ids} for id {sid} and task {task}")
    return True

def main():
    args = parse_args()

    validate_lists(args.subject_ids, args.tasks)
    comfi_root = Path(args.comfi_root)

    total = 0
    ok = 0
    for sid in args.subject_ids:
        for task in args.tasks:
            total += 1
            ok += int(process_one(comfi_root, sid, task, args.show_realtime, args.cam_ids))

    print(f"[DONE] {ok}/{total} combinations processed.")

if __name__ == "__main__":
    main()
