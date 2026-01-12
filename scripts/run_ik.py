# This script solves an inverse kinematics problem using QP or IPOPT to estimate joint angles from LSTM-augmented anatomical marker data.
import os
import sys
from comfi_examples.urdf_utils import *
import numpy as np
import pinocchio as pin
from comfi_examples.utils import read_mks_data, read_subject_yaml, Rquat
import pandas as pd
from  comfi_examples.ik_utils import RT_IK


nbr_cams= 2
mks_to_skip = ['LForearm','LUArm', 'RUArm', 'RHJC_study','LHJC_study','r_pelvis','l_pelvis',
               'LHand','LHL2','LHM5', 'RForearm','RHand','RHL2','RHM5', 'L_sh1_study', 'L_thigh1_study','r_sh1_study', 'r_thigh1_study',
               'r_thigh2_study', 'L_sh2_study', 'L_thigh2_study','r_sh2_study', 
               'L_sh3_study', 'L_thigh3_study','r_sh3_study', 'r_thigh3_study']

#read mks data

path_to_csv = f"/home/kchalabi/Documents/THESE/dataset/comfi-usage/output/res_hpe/1012/RobotWelding/augmented_markers.csv"
path_to_kpt = f"/home/kchalabi/Documents/THESE/dataset/comfi-usage/output/res_hpe/1012/RobotWelding/3d_keypoints_4cams.csv"
keys_to_add =  ['Nose', 'Head', 'Right_Ear', 'Left_Ear', 'Right_Eye', 'Left_Eye']

data_markers_lstm = pd.read_csv(path_to_csv) 
keypoints = pd.read_csv(path_to_kpt)/1000

columns_to_add = [col for col in keypoints.columns if any(key + '_' in col for key in keys_to_add)]

mks_data = pd.concat([data_markers_lstm, keypoints[columns_to_add].reset_index(drop=True)], axis=1)

###################################################################""""
start_sample=0
result_markers, start_sample_dict = read_mks_data(mks_data, start_sample=start_sample) #check the function of read 
print(start_sample_dict)
#load urdf
mesh_dir= "/home/kchalabi/Documents/THESE/dataset/comfi-usage/model"
urdf_path = "/home/kchalabi/Documents/THESE/dataset/comfi-usage/model/urdf/human.urdf"
human = Robot(urdf_path,mesh_dir,isFext=True) 
human_model = human.model
human_data = human.data
human_collision_model = human.collision_model
human_visual_model = human.visual_model

metadata_yaml = "/home/kchalabi/Documents/THESE/dataset/comfi-usage/COMFI/metadata/1012.yaml"
_, subject_height, subject_mass, gender = read_subject_yaml(str(metadata_yaml))

#scale the model to data
human_model = scale_human_model(human_model, start_sample_dict,with_hand=True,gender=gender,subject_height=subject_height)
print(human_model.nq)
human_model= mks_registration(human_model,start_sample_dict, with_hand=False)
human_data = pin.Data(human_model)

################################################################################LOCK JOINTS
all_joint_ids = set(range(1, human_model.njoints))
joints_to_lock = ["middle_thoracic_X", "middle_thoracic_Y", "middle_thoracic_Z", "left_wrist_X", "left_wrist_Z", "right_wrist_X","right_wrist_Z"]
joint_ids_to_lock = []
for jn in joints_to_lock:
    if human_model.existJointName(jn):
        joint_ids_to_lock.append(human_model.getJointId(jn))
    else:
        print('Warning: joint ' + str(jn) + ' does not belong to the model!')

q0 = pin.neutral(human_model)
# Build reduced model
human_model, human_visual_model = pin.buildReducedModel(
    human_model, human_visual_model, joint_ids_to_lock, q0)

print(human_model.nq)
human_data = pin.Data(human_model)
###############################################################################################################



### IK init 
q = pin.neutral(human_model) # init pos
human_data = pin.Data(human_model)

dt = 1/40 #dt for qp
keys_to_track_list = ['FHD', 'LHD', 'RHD',  
        'C7_study', 
        'r.ASIS_study', 'L.ASIS_study', 
        'r.PSIS_study', 'L.PSIS_study', 
        'r_shoulder_study',
        'r_lelbow_study', 'r_melbow_study',
        'r_lwrist_study', 'r_mwrist_study',
        'r_ankle_study', 'r_mankle_study',
        'r_toe_study','r_5meta_study', 'r_calc_study',
        'r_knee_study', 'r_mknee_study',
        'L_shoulder_study', 
        'L_lelbow_study', 'L_melbow_study',
        'L_lwrist_study','L_mwrist_study',
        'L_ankle_study', 'L_mankle_study', 
        'L_toe_study','L_5meta_study', 'L_calc_study',
        'L_knee_study', 'L_mknee_study',
                        ]


omega = {}
for key in keys_to_track_list:
    omega[key] = 1

### IK calculations
ik_class = RT_IK(human_model, start_sample_dict, q, keys_to_track_list, dt,omega)
q = ik_class.solve_ik_sample_casadi() #warm start with ipopt for qp 
ik_class._q0=q

rmse_per_marker = {}
q_list = []
M_model_list = []

for ii in range(start_sample,len(result_markers)): 

    mks_dict = result_markers[ii]
    ik_class._dict_m= mks_dict
    q = ik_class.solve_ik_sample_casadi() 

    pin.forwardKinematics(human_model, human_data, q)
    pin.updateFramePlacements(human_model, human_data)
    
    M_model_frame = {}

    for marker in result_markers[ii].keys():
        # print(marker)
        if marker in mks_to_skip: 
            continue  #skip
        pos_gt = np.array(result_markers[ii][marker])

        M = pin.SE3(pin.SE3(Rquat(1, 0, 0, 0), np.matrix([result_markers[ii][marker][0],result_markers[ii][marker][1],result_markers[ii][marker][2]]).T))
        M_model = human_data.oMf[human_model.getFrameId(marker)]
        pos_model = np.array(M_model.translation).flatten()


        # Add marker_model position to the frame's data
        M_model_frame[f"{marker}_x"] = M_model.translation[0]
        M_model_frame[f"{marker}_y"] = M_model.translation[1]
        M_model_frame[f"{marker}_z"] = M_model.translation[2]
        


        # RMSE calculation
        sq_error = np.sum((pos_gt - pos_model) ** 2)

        if marker not in rmse_per_marker:
            rmse_per_marker[marker] = []
        rmse_per_marker[marker].append(sq_error)

    M_model_list.append(M_model_frame)
    
    # Display frames from measurements
    # seg_frames = construct_segments_frames(mks_dict)
    # for seg_name, M in seg_frames.items():
        
    #     frame_name = f'world/{seg_name+"_meas"}'
    #     frame_se3 = pin.SE3(M[:3,:3], np.matrix([M[0,3],M[1,3],M[2,3]]).T)
    #     place(viz, frame_name, frame_se3)
    pin.forwardKinematics(human_model,human_data, q)
    pin.updateFramePlacements(human_model,human_data)
    # #  Display frames from human_model
    # for joint_id in range(1, human_model.njoints):  # Skip 0 (universe)
    #     frame_name = f'world/{human_model.names[joint_id]+"_model"}'
    #     frame_se3= human_data.oMf[human_model.getFrameId(human_model.names[joint_id])]
    #     place(viz, frame_name, frame_se3)

    #display q
    # input("Press Enter to continue...")
    ik_class._q0 = q 

    q_list.append(q)

#save mks est
# df = pd.DataFrame(M_model_list)
# csv_file = os.path.join(rt_cosmik_path,f"/root/workspace/ros_ws/src/rt-cosmik/output/{no_trial}/{task}/mks_model_cosmik_ipopt.csv") 
# df.to_csv(csv_file, index=False)

#save angles
joint_angles_names = ['FF_X', 'FF_Y', 'FF_Z', 'FF_quatx','FF_quaty',
                          'FF_quatz', 'FF_quatw', 'Lhip_flex_ext', 'Lhip_abd_add','Lhip_int_ext_rot','Lknee_flex_ext','Lankle_flex_ext','Lankle_abd_add',
                          'Lumbar_flex_ext', 'Lumbar_lateral_flex',
                          'Lcalvicule_x',
                          'Lshoulder_flex_ext','Lshoulder_abd_add', 'Lshoulder_int_ext_rot','Lelbow_flex_ext','Lelbow_pron_supi',
                          'Cervical_flex_ext', 'Cervical_lat_bend', 'Cervical_int_ext_rot',
                          'rcalvicule_x',
                          'Rshoulder_flex_ext', 'Rshoulder_abd_add', 'Rshoulder_int_ext_rot','Relbow_flex_ext', 'Relbow_pron_supi', 
                          'Rhip_flex_ext','Rhip_abd_add','Rhip_int_ext_rot',
                          'Rknee_flex_ext','Rankle_flex_ext', 'Rankle_abd_add']
num_values = len(q_list[0])
if len(joint_angles_names) != num_values:
    raise ValueError(f"joint_angles_names has {len(joint_angles_names)} entries but q has {num_values} DOFs.")

df = pd.DataFrame(q_list, columns=joint_angles_names)
# csv_file = os.path.join(rt_cosmik_path, f"output/{id}/cosmik_2cams/{task}/q_cosmik_mocap_finetunednew.csv")
# df.to_csv(csv_file, index=False)



rmse_global = 0
nb_mks =0 
# Final RMSE output
print("\nPer-marker RMSE (in meters):")
for marker, sq_errors in rmse_per_marker.items():
    nb_mks +=1
    rmse = np.sqrt(np.mean(sq_errors))
    print(f"{marker}: {rmse:.4f} m")
    rmse_global +=rmse

rmse_global = rmse_global/nb_mks
print(f" Global RMSE across all markers and frames: {rmse_global:.4f} m")

