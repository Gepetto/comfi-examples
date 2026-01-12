from pinocchio.robot_wrapper import RobotWrapper
import pinocchio as pin
import numpy as np 
from typing import List, Tuple, Dict
from comfi_examples.linear_algebra_utils import col_vector_3D

from comfi_examples.linear_algebra_utils import (
    transform_to_global_frame,
    transform_to_local_frame,
    orthogonalize_matrix,
)

def get_head_pose(mks_positions):
    """
    Calculate the pose of the head based on motion capture marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the head.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                Expected keys are 'Neck', 'midHip', 'C7_study', 'CV7', 'SJN', 
                                'HeadR', 'HeadL', 'RSAT', and 'LSAT'. Each key should map to a 
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the head pose.
    """

    pose = np.eye(4,4)
    X, Y, Z, head_center = [], [], [], []
    if 'Head' in mks_positions:
        head_center = (mks_positions['RSHO'] + mks_positions['LSHO'])/2.0 
        top_head = mks_positions['Head']
        Y = (top_head - head_center).reshape(3,1)
        Y = Y/np.linalg.norm(Y)

        Z = (mks_positions['Right_Ear'] - mks_positions['Left_Ear']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    elif 'BHD' in mks_positions: 
        head_center = (mks_positions['RSHO'] + mks_positions['LSHO'])/2.0 
        top_head = (mks_positions['FHD'] +
                mks_positions['BHD'] +
                mks_positions['LHD'] +
                mks_positions['RHD'] )/4.0
        Y = (top_head - head_center).reshape(3,1)
        Y = Y/np.linalg.norm(Y)

        Z = mks_positions['RHD'] - mks_positions['LHD']
        Z = Z/np.linalg.norm(Z)

        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)
    else: 
        head_center = (mks_positions['RSHO'] + mks_positions['LSHO'])/2.0 
        top_head = (mks_positions['FHD'] +
                mks_positions['LHD'] +
                mks_positions['RHD'] )/3.0
        Y = (top_head - head_center).reshape(3,1)
        Y = Y/np.linalg.norm(Y)

        Z = mks_positions['RHD'] - mks_positions['LHD']
        Z = Z/np.linalg.norm(Z)

        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = head_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

# construct torso frame and get its pose from a dictionnary of mks positions and names
def get_torso_pose(mks_positions):
    """
    Calculate the torso pose matrix from motion capture marker positions.
    The function computes a 4x4 transformation matrix representing the pose of the torso.
    The matrix includes rotation and translation components derived from the positions
    of specific markers.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                Expected keys are 'Neck', 'Mid_Hip', 'C7', 'CV7', 'SJN',
                                'HeadR', 'HeadL', 'RSAT', and 'LSAT'. Each key should map to a
                                numpy array of shape (3,).
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the torso pose.
    """

    pose = np.eye(4, 4)
    X, Y, Z, trunk_center = [], [], [], []

    trunk_center = (mks_positions["RSHO"] + mks_positions["LSHO"]) / 2.0
    Mid_Hip = (
        mks_positions["RASI"]
        + mks_positions["LASI"]
        + mks_positions["RPSI"]
        + mks_positions["LPSI"]
    ) / 4.0

    Y = (trunk_center - Mid_Hip).reshape(3, 1)
    Y = Y / np.linalg.norm(Y)
    X = (trunk_center - mks_positions["C7"]).reshape(3, 1)
    X = X / np.linalg.norm(X)

    Z = np.cross(X, Y, axis=0)
    X = np.cross(Y, Z, axis=0)

    pose[:3, 0] = X.reshape(
        3,
    )
    pose[:3, 1] = Y.reshape(
        3,
    )
    pose[:3, 2] = Z.reshape(
        3,
    )
    pose[:3, 3] = trunk_center.reshape(
        3,
    )
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])
    return pose

#construct upperarm frames and get their poses
def get_upperarmR_pose(mks_positions):
    """
    Calculate the pose of the right upper arm based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RShoulder', 'r_melbow_study', 'r_lelbow_study', 
                                'RHLE', 'RHME', 'RSAT', and 'LSAT'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right upper arm. 
                   The matrix includes rotation (3x3) and translation (3x1) components.
    """

    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []

    torso_pose = get_torso_pose(mks_positions)
    bi_acromial_dist = np.linalg.norm(mks_positions['LSHO'] - mks_positions['RSHO'])
    shoulder_center = mks_positions['RSHO'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3)) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)
    elbow_center = (mks_positions['RMELB'] + mks_positions['RELB']).reshape(3,1)/2.0
    
    Y = shoulder_center - elbow_center
    Y = Y/np.linalg.norm(Y)

    Z = (mks_positions['RELB'] - mks_positions['RMELB']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

        
    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = shoulder_center.reshape(3,)

    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose


def get_upperarmL_pose(mks_positions):
    """
    Calculate the pose of the left upper arm based on motion capture marker positions.
    This function computes the transformation matrix representing the pose of the left upper arm.
    The pose is calculated using the positions of specific markers on the body, such as the shoulder
    and elbow markers. The resulting pose matrix is a 4x4 homogeneous transformation matrix.
    Args:
        mks_positions (dict): A dictionary containing the positions of motion capture markers.
            The keys are marker names (e.g., 'LShoulder', 'L_melbow_study', 'L_lelbow_study', 'LHLE', 'LHME', 'LSAT', 'RSAT'),
            and the values are numpy arrays of shape (3,) representing the 3D coordinates of the markers.
    Returns:
        numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left upper arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, shoulder_center = [], [], [], []
    torso_pose = get_torso_pose(mks_positions)
    bi_acromial_dist = np.linalg.norm(mks_positions['LSHO'] - mks_positions['RSHO'])
    shoulder_center = mks_positions['LSHO'].reshape(3,1) + (torso_pose[:3, :3].reshape(3,3) @ col_vector_3D(0.0, -0.17*bi_acromial_dist, 0.0)).reshape(3,1)
    elbow_center = (mks_positions['LMELB'] + mks_positions['LELB']).reshape(3,1)/2.0
    
    Y = shoulder_center - elbow_center
    Y = Y/np.linalg.norm(Y)

    Z = (mks_positions['LMELB'] - mks_positions['LELB']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3, 0] = X.flatten()
    pose[:3, 1] = Y.flatten()
    pose[:3, 2] = Z.flatten()
    pose[:3, 3] = shoulder_center.flatten()
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])


    # print("Upperarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité
 
    # print("Upperarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité

    return pose


#construct lowerarm frames and get their poses
def get_lowerarmR_pose(mks_positions):
    """
    Calculate the pose of the right lower arm based on motion capture marker positions.
    The function computes the transformation matrix (pose) of the right lower arm using the positions of specific markers.
    It first checks for the presence of 'r_melbow_study' in the marker positions to determine which set of markers to use.
    The pose is represented as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. The keys are marker names,
                                and the values are their corresponding 3D positions (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the right lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    elbow_center = (mks_positions['RMELB'] + mks_positions['RELB']).reshape(3,1)/2.0
    wrist_center = (mks_positions['RMWRI'] + mks_positions['RWRI']).reshape(3,1)/2.0
    
    Y = elbow_center - wrist_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RWRI'] - mks_positions['RMWRI']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = elbow_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_lowerarmL_pose(mks_positions):
    """
    Calculate the pose of the left lower arm based on motion capture marker positions.
    This function computes the transformation matrix representing the pose of the left lower arm.
    It uses the positions of specific markers to determine the orientation and position of the arm.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                The keys should include either 'L_melbow_study', 'L_lelbow_study', 
                                'L_mwrist_study', 'L_lwrist_study' or 'LHLE', 'LHME', 'LRSP', 'LUSP'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left lower arm.
    """

    pose = np.eye(4,4)
    X, Y, Z, elbow_center = [], [], [], []
    elbow_center = (mks_positions['LMELB'] + mks_positions['LELB']).reshape(3,1)/2.0
    wrist_center = (mks_positions['LMWRI'] + mks_positions['LWRI']).reshape(3,1)/2.0
    
    Y = elbow_center - wrist_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['LMWRI'] - mks_positions['LWRI']).reshape(3,1)
    # Z = Z/np.linalg.norm(Z)
    Z = Z.reshape(3, 1) / np.linalg.norm(Z)

    X = np.cross(Y, Z, axis=0)
    X = X.reshape(3, 1) / np.linalg.norm(X)

    Z = np.cross(X.flatten(), Y.flatten())
    Z = Z.reshape(3, 1) / np.linalg.norm(Z)
    # Z = np.cross(X, Y, axis=0)


    pose[:3, 0] = X.flatten()
    pose[:3, 1] = Y.flatten()
    pose[:3, 2] = Z.flatten()
    pose[:3, 3] = elbow_center.flatten()
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])


    # print("Lowerarm Left Pose:\n", pose)  # Impression pour débogage
    # check_orthogonality(pose)  # Ajoutez cette ligne pour vérifier l'orthogonalité

    return pose

#construc hand frame and get its pose
def get_handR_pose(mks_positions):
    """
    Calculate the pose of the right hand based on motion capture marker positions.
    The function computes the transformation matrix (pose) of the right hand  using the positions of specific markers.
    It first checks for the presence of 'r_melbow_study' in the marker positions to determine which set of markers to use.
    The pose is represented as a 4x4 homogeneous transformation matrix.
    Parameters:f
    mks_positions (dict): A dictionary containing the positions of motion capture markers. The keys are marker names,
                                and the values are their corresponding 3D positions (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the right hand .
    """

    pose = np.eye(4,4)
    X, Y, Z, wrist_center = [], [], [], []
    
    if 'RHL2' in mks_positions:
        wrist_center = (mks_positions['RMWRI'] + mks_positions['RWRI']).reshape(3,1)/2.0
        metacarpal_center = (mks_positions['RHL2'] + mks_positions['RHM5']).reshape(3,1)/2.0
        
        Y = wrist_center - metacarpal_center
        Y = Y/np.linalg.norm(Y)
        Z = (mks_positions['RHL2'] - mks_positions['RHM5']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = wrist_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
        return pose
    else:
        elbow_center = (mks_positions['RMELB'] + mks_positions['RELB']).reshape(3,1)/2.0
        wrist_center = (mks_positions['RMWRI'] + mks_positions['RWRI']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mks_positions['RWRI'] - mks_positions['RMWRI']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = wrist_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
        return pose

def get_handL_pose(mks_positions):
    """
    Calculate the pose of the left hand based on motion capture marker positions.
    The function computes the transformation matrix (pose) of the left hand using the positions of specific markers.
    It first checks for the presence of 'r_melbow_study' in the marker positions to determine which set of markers to use.
    The pose is represented as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. The keys are marker names,
                                and the values are their corresponding 3D positions (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left hand.
    """

    pose = np.eye(4,4)
    X, Y, Z, wrist_center = [], [], [], []
    if 'LHL2' in mks_positions:
        wrist_center = (mks_positions['LMWRI'] + mks_positions['LWRI']).reshape(3,1)/2.0
        metacarpal_center = (mks_positions['LHL2'] + mks_positions['LHM5']).reshape(3,1)/2.0
        
        Y = wrist_center - metacarpal_center
        Y = Y/np.linalg.norm(Y)
        Z = (mks_positions['LHM5'] - mks_positions['LHL2']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = wrist_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
        return pose
    else:
        elbow_center = (mks_positions['LMELB'] + mks_positions['LELB']).reshape(3,1)/2.0
        wrist_center = (mks_positions['LMWRI'] + mks_positions['LWRI']).reshape(3,1)/2.0
        
        Y = elbow_center - wrist_center
        Y = Y/np.linalg.norm(Y)
        Z = (mks_positions['LWRI'] - mks_positions['LMWRI']).reshape(3,1)
        Z = Z/np.linalg.norm(Z)
        X = np.cross(Y, Z, axis=0)
        Z = np.cross(X, Y, axis=0)

        pose[:3,0] = X.reshape(3,)
        pose[:3,1] = Y.reshape(3,)
        pose[:3,2] = Z.reshape(3,)
        pose[:3,3] = wrist_center.reshape(3,)
        pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
        return pose


def force_vertical_y_rotation(R_initial: np.ndarray, y_vertical: np.ndarray) -> np.ndarray:
    """
    Reconstruct a rotation matrix from R_initial, forcing the Y-axis to match y_vertical.
    
    Parameters:
    - R_initial (np.ndarray): 3x3 rotation matrix.
    - y_vertical (np.ndarray): 3D unit vector representing the desired vertical Y-axis.

    Returns:
    - R_new (np.ndarray): 3x3 rotation matrix with Y-axis aligned to y_vertical.

    Raises:
    - ValueError if the resulting matrix is not a proper rotation (det ≠ 1)
    """
    y_new = y_vertical / np.linalg.norm(y_vertical)
    z_known = R_initial[:, 1:2]

    # Try to construct a valid x_new
    x_new = np.cross(z_known.flatten(), y_new.flatten(),axis=0)
    
    if np.linalg.norm(x_new) < 1e-6:
        # z and y are nearly aligned → fallback to use original x axis
        x_known = R_initial[:, 0]
        x_new = np.cross(x_known, y_new)

    x_new = x_new / np.linalg.norm(x_new)
    z_new = np.cross(x_new.flatten(), y_new.flatten())

    R_new = np.column_stack((x_new, y_new, z_new))

    # Check determinant
    det = np.linalg.det(R_new)
    if not np.isclose(det, 1.0, atol=1e-6):
        raise ValueError(f"Invalid rotation matrix: det = {det}, expected 1.0")

    return R_new

#construct abdomen frame and get its pose (middle thoracic joint in urdf)
def get_thorax_pose(mks_positions,gender='male',subject_height= 1.80):
    #pelvis + distance selon y
    """
    Calculate the abdomen pose matrix from motion capture marker positions.
    The function computes the abdomen pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the abdomen coordinate system. Finally, it constructs the 
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'r.PSIS_study', 'L.PSIS_study', 'r.ASIS_study', 
                                'L.ASIS_study', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the abdomen pose.
    """
    if gender == 'male' : 
        abdomen_ratio = 0.0839
    else : 
        abdomen_ratio = 0.0776
    
    pelvis_position =(get_pelvis_pose(mks_positions,gender)[:3,3]).reshape(3,1)
    torso_position = (get_torso_pose(mks_positions)[:3,3]).reshape(3,1)
    vertical_direction = torso_position - pelvis_position                    
    vertical_direction = vertical_direction / np.linalg.norm(vertical_direction)  

    trans_local=col_vector_3D(0.0, subject_height * abdomen_ratio,0.0)
    pelvis_pose_forced = force_vertical_y_rotation(get_pelvis_pose(mks_positions,gender)[:3,:3].reshape(3,3), vertical_direction)
    # trans_global = (pelvis_pose_forced@ trans_local).reshape(3,1)
    trans_global = (get_pelvis_pose(mks_positions,gender)[:3,:3].reshape(3,3) @ trans_local).reshape(3,1)

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    center_PSIS = []
    center_ASIS = []

    center_PSIS = (mks_positions['RPSI'] + mks_positions['LPSI']).reshape(3,1)/2.0
    center_ASIS = (mks_positions['RASI'] + mks_positions['LASI']).reshape(3,1)/2.0

    center_right_ASIS_PSIS = (mks_positions['RPSI'] + mks_positions['RASI']).reshape(3,1)/2.0
    center_left_ASIS_PSIS = (mks_positions['LPSI'] + mks_positions['LASI']).reshape(3,1)/2.0
    
    X = center_ASIS - center_PSIS
    X = X/np.linalg.norm(X)
    # Z = mks_positions['r.ASIS_study'] - mks_positions['L.ASIS_study']
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ((get_pelvis_pose(mks_positions,gender)[:3,3]).reshape(3,1)+ trans_global).reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])

    return pose

# get_virtual_pelvis_pose, used to get thigh pose
def get_virtual_pelvis_pose(mks_positions):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the
    pose matrix and ensures it is orthogonal.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'RPSI', 'LPSI', 'RASI',
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    pose = np.eye(4, 4)
    X, Y, Z = [], [], []
    center_PSIS = []
    center_ASIS = []

    center_PSIS = (mks_positions["RPSI"] + mks_positions["LPSI"]).reshape(3, 1) / 2.0
    center_ASIS = (mks_positions["RASI"] + mks_positions["LASI"]).reshape(3, 1) / 2.0
    _center = (
        mks_positions["RASI"]
        + mks_positions["LASI"]
        + mks_positions["RPSI"]
        + mks_positions["LPSI"]
    ) / 4.0
    X = center_ASIS - center_PSIS
    X = X / np.linalg.norm(X)
    Z = mks_positions["RASI"] - mks_positions["LASI"]
    Z = Z / np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3, 0] = X.reshape(
        3,
    )
    pose[:3, 1] = Y.reshape(
        3,
    )
    pose[:3, 2] = Z.reshape(
        3,
    )
    pose[:3, 3] = center_ASIS.reshape(
        3,
    )
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])
    return pose


def get_pelvis_pose(mks_positions, gender="male"):
    """
    Calculate the pelvis pose matrix from motion capture marker positions.
    The function computes the pelvis pose based on the positions of specific markers.
    It first determines the center points of the PSIS and ASIS markers, then calculates
    the X, Y, and Z axes of the pelvis coordinate system. Finally, it constructs the
    pose matrix and ensures it is orthogonal.
    Parameters:
    mocap_mks_positions (dict): A dictionary containing the positions of the motion capture markers.
                                The keys can be either 'RPSI', 'LPSI', 'RASI',
                                'LASI', or 'RIPS', 'LIPS', 'RIAS', 'LIAS'.
    Returns:
    numpy.ndarray: A 4x4 pose matrix representing the pelvis pose.
    """

    if gender == "male":
        ratio_x = 0.335
        ratio_y = -0.032
        ratio_z = 0.0
    else:
        ratio_x = 0.34
        ratio_y = 0.049
        ratio_z = 0.0

    pose = np.eye(4, 4)
    center_PSIS = []
    center_ASIS = []
    center_right_ASIS_PSIS = []
    center_left_ASIS_PSIS = []
    LJC = np.zeros((3, 1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["RASI"] - mks_positions["LASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    LJC = virtual_pelvis_pose[:3, 3].reshape(3, 1)

    center_PSIS = (mks_positions["RPSI"] + mks_positions["LPSI"]).reshape(3, 1) / 2.0
    center_ASIS = (mks_positions["RASI"] + mks_positions["LASI"]).reshape(3, 1) / 2.0

    center_right_ASIS_PSIS = (mks_positions["RPSI"] + mks_positions["RASI"]).reshape(
        3, 1
    ) / 2.0
    center_left_ASIS_PSIS = (mks_positions["LPSI"] + mks_positions["LASI"]).reshape(
        3, 1
    ) / 2.0

    offset_local = col_vector_3D(
        -ratio_x * dist_rPL_lPL, +ratio_y * dist_rPL_lPL, ratio_z * dist_rPL_lPL
    )
    LJC = LJC + virtual_pelvis_pose[:3, :3] @ offset_local

    X = center_ASIS - center_PSIS
    X = X / np.linalg.norm(X)
    # Z = mks_positions['RASI'] - mks_positions['LASI']
    Z = center_right_ASIS_PSIS - center_left_ASIS_PSIS
    Z = Z / np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3, 0] = X.reshape(
        3,
    )
    pose[:3, 1] = Y.reshape(
        3,
    )
    pose[:3, 2] = Z.reshape(
        3,
    )
    pose[:3, 3] = ((center_right_ASIS_PSIS + center_left_ASIS_PSIS) / 2.0).reshape(
        3,
    )
    # pose[:3,3] = LJC.reshape(3,)
    pose[:3, :3] = orthogonalize_matrix(pose[:3, :3])

    return pose

#construct thigh frames and get their poses
def get_thighR_pose(mks_positions, gender='male'):
    """
    Calculate the pose of the right thigh based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys include 'RHip', 'r_knee_study', 'r_mknee_study', 
                                'RIAS', 'LIAS', 'RFLE', and 'RFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right thigh. The matrix 
                   includes rotation and translation components.
    """
    if gender == 'male':
        ratio_x = 0.3
        ratio_y = 0.37
        ratio_z = 0.361
    else : 
        ratio_x = 0.3
        ratio_y = 0.336
        ratio_z = 0.372

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["RASI"]-mks_positions["LASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    hip_center = virtual_pelvis_pose[:3, 3].reshape(3,1)

    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-ratio_x*dist_rPL_lPL, 0.0, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -ratio_y*dist_rPL_lPL, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, ratio_z*dist_rPL_lPL)

    knee_center = (mks_positions['RKNE'] + mks_positions['RMKNE']).reshape(3,1)/2.0
    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RKNE'] - mks_positions['RMKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose


def get_thighL_pose(mks_positions, gender='male'):
    """
    Calculate the pose of the left thigh based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                Expected keys are 'LHip', 'L_knee_study', 'L_mknee_study', 'LIAS', 'RIAS', 'LFLE', and 'LFME'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left thigh. The matrix includes
                   rotation and translation components.
    """
    if gender == 'male':
        ratio_x = 0.3
        ratio_y = 0.37
        ratio_z = 0.361
    else : 
        ratio_x = 0.3
        ratio_y = 0.336
        ratio_z = 0.372

    pose = np.eye(4,4)
    X, Y, Z = [], [], []
    hip_center = np.zeros((3,1))

    dist_rPL_lPL = np.linalg.norm(mks_positions["LASI"]-mks_positions["RASI"])
    virtual_pelvis_pose = get_virtual_pelvis_pose(mks_positions)
    hip_center = virtual_pelvis_pose[:3, 3].reshape(3,1)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(-ratio_x*dist_rPL_lPL, 0.0, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, -ratio_y*dist_rPL_lPL, 0.0)
    hip_center = hip_center + virtual_pelvis_pose[:3,:3].reshape(3,3) @ col_vector_3D(0.0, 0.0, -ratio_z*dist_rPL_lPL)

    knee_center = (mks_positions['LKNE'] + mks_positions['LMKNE']).reshape(3,1)/2.0
    Y = hip_center - knee_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['LMKNE'] - mks_positions['LKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)

    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = hip_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct shank frames and get their poses
def get_shankR_pose(mks_positions):
    """
    Calculate the pose of the right shank based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'r_knee_study', 'r_mknee_study', 
                                'r_mankle_study', 'r_ankle_study' or 'RFLE', 'RFME', 'RTAM', 'RFAL'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right shank. The matrix 
                   includes rotation (in the top-left 3x3 submatrix) and translation (in the top-right 
                   3x1 subvector).
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []

    knee_center = (mks_positions['RKNE'] + mks_positions['RMKNE']).reshape(3,1)/2.0
    ankle_center = (mks_positions['RMANK'] + mks_positions['RANK']).reshape(3,1)/2.0
    Y = knee_center - ankle_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['RKNE'] - mks_positions['RMKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_shankL_pose(mks_positions):
    """
    Calculate the pose of the left shank based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys should include either 'L_knee_study', 'L_mknee_study', 
                                'L_mankle_study', 'L_ankle_study' or 'LFLE', 'LFME', 'LTAM', 'LFAL'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the left shank. The matrix 
                   includes the rotation (3x3) and translation (3x1) components.
    """

    pose = np.eye(4,4)
    X, Y, Z, knee_center, ankle_center = [], [], [], [], []

    knee_center = (mks_positions['LKNE'] + mks_positions['LMKNE']).reshape(3,1)/2.0
    ankle_center = (mks_positions['LMANK'] + mks_positions['LANK']).reshape(3,1)/2.0
    Y = knee_center - ankle_center
    Y = Y/np.linalg.norm(Y)
    Z = (mks_positions['LMKNE'] - mks_positions['LKNE']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    X = np.cross(Y, Z, axis=0)
    Z = np.cross(X, Y, axis=0)


    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = knee_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#construct foot frames and get their poses
def get_footR_pose(mks_positions):
    """
    Calculate the pose of the right foot based on motion capture marker positions.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers. 
                                The keys can be either 'r_mankle_study', 'r_ankle_study', 'r_toe_study', 
                                'r_calc_study' or 'RTAM', 'RFAL', 'RFM5', 'RFM1', 'RFCC'.
    Returns:
    numpy.ndarray: A 4x4 transformation matrix representing the pose of the right foot. The matrix 
                   includes the orientation (rotation) and position (translation) of the foot.
    """

    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []

    ankle_center = (mks_positions['RMANK'] + mks_positions['RANK']).reshape(3,1)/2.0
    toe_pos = (mks_positions['RTOE'] + mks_positions['R5MHD'])/2.0
    
    X = (toe_pos - mks_positions['RHEE']).reshape(3,1)  
    X = X/np.linalg.norm(X)
    Z = (mks_positions['RANK'] - mks_positions['RMANK']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)




    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

def get_footL_pose(mks_positions):
    """
    Calculate the pose of the left foot based on motion capture marker positions.
    This function computes the transformation matrix (pose) of the left foot using
    the positions of various markers from motion capture data. The pose is represented
    as a 4x4 homogeneous transformation matrix.
    Parameters:
    mks_positions (dict): A dictionary containing the positions of motion capture markers.
                                The keys are marker names and the values are their respective
                                3D coordinates (numpy arrays).
    Returns:
    numpy.ndarray: A 4x4 homogeneous transformation matrix representing the pose of the left foot.
    Notes:
    - The function checks for the presence of specific markers ('L_mankle_study', 'L_ankle_study',
      'L_toe_study', 'L_calc_study') to compute the pose. If these markers are not present, it
      uses alternative markers ('LTAM', 'LFAL', 'LFM5', 'LFM1', 'LFCC').
    - The resulting pose matrix includes the orientation (rotation) and position (translation)
      of the left foot.
    - The orientation matrix is orthogonalized to ensure it is a valid rotation matrix.
    """

    pose = np.eye(4,4)
    X, Y, Z, ankle_center = [], [], [], []

    ankle_center = (mks_positions['LMANK'] + mks_positions['LANK']).reshape(3,1)/2.0
    toe_pos = (mks_positions['LTOE'] + mks_positions['L5MHD'])/2.0

    X = (toe_pos - mks_positions['LHEE']).reshape(3,1)
    X = X/np.linalg.norm(X)
    Z = (mks_positions['LMANK'] - mks_positions['LANK']).reshape(3,1)
    Z = Z/np.linalg.norm(Z)
    Y = np.cross(Z, X, axis=0)
    Z = np.cross(X, Y, axis=0)



    pose[:3,0] = X.reshape(3,)
    pose[:3,1] = Y.reshape(3,)
    pose[:3,2] = Z.reshape(3,)
    pose[:3,3] = ankle_center.reshape(3,)
    pose[:3,:3] = orthogonalize_matrix(pose[:3,:3])
    return pose

#Construct challenge segments frames from mocap mks
# - mks_positions is a dictionnary of mocap mks names and 3x1 global positions
# - returns sgts_poses which correspond to a dictionnary to segments poses and names, constructed from mks global positions
def construct_segments_frames(mks_positions, with_hand=True, with_head=True, gender='male',subject_height=1.8): 
    """
    Constructs a dictionary of segment poses from motion capture marker positions.
    Args:
        mks_positions (dict): A dictionary containing the positions of motion capture markers.
    Returns:
        dict: A dictionary where keys are segment names (e.g., 'torso', 'upperarmR') and values are the corresponding poses.
    """

    torso_pose = get_torso_pose(mks_positions)
    right_clavicle_pose =get_torso_pose(mks_positions)
    left_clavicle_pose =get_torso_pose(mks_positions)
    upperarmR_pose = get_upperarmR_pose(mks_positions)
    lowerarmR_pose = get_lowerarmR_pose(mks_positions)
    upperarmL_pose = get_upperarmL_pose(mks_positions)
    lowerarmL_pose = get_lowerarmL_pose(mks_positions) 
    thorax_pose = get_thorax_pose(mks_positions,gender='male',subject_height=1.8)
    pelvis_pose = get_pelvis_pose(mks_positions,gender='male')
    thighR_pose = get_thighR_pose(mks_positions,gender='male')
    shankR_pose = get_shankR_pose(mks_positions)
    footR_pose = get_footR_pose(mks_positions)
    thighL_pose = get_thighL_pose(mks_positions,gender='male')
    shankL_pose = get_shankL_pose(mks_positions)
    footL_pose = get_footL_pose(mks_positions)
    
    # Constructing the dictionary to store segment poses
    sgts_poses = {
        # "head": head_pose,
        "torso": torso_pose,
        "right_clavicle" : right_clavicle_pose,
        "left_clavicle" : left_clavicle_pose,
        "upperarmR": upperarmR_pose,
        "lowerarmR": lowerarmR_pose,
        "upperarmL": upperarmL_pose,
        "lowerarmL": lowerarmL_pose,
        "pelvis": pelvis_pose,
        "thorax":thorax_pose,
        "thighR": thighR_pose,
        "shankR": shankR_pose,
        "footR": footR_pose,
        "thighL": thighL_pose,
        "shankL": shankL_pose,
        "footL": footL_pose
    }
    if with_head :
        head_pose = get_head_pose(mks_positions)
        sgts_poses["head"] = head_pose
    
    if with_hand : 
        handR_pose = get_handR_pose(mks_positions)
        handL_pose = get_handL_pose(mks_positions)
        sgts_poses["handR"] = handR_pose
        sgts_poses["handL"] = handL_pose

    # for name, pose in sgts_poses.items():
    #     print(name, " rot det : ", np.linalg.det(pose[:3,:3]))
    return sgts_poses

def get_local_segments_positions(sgts_poses: Dict, with_hand=True)->Dict:
    """_Get the local positions of the segments_

    Args:
        sgts_poses (Dict): _a dictionnary of segment poses_

    Returns:
        Dict: _returns a dictionnary of local positions for each segment except pelvis_
    """
    # Initialize the dictionary to store local positions
    local_positions = {}

    # Pelvis is the base, so it does not have a local position
    pelvis_pose = sgts_poses["pelvis"]
    # Compute local positions for each segment
    
    if "thorax" in sgts_poses:
        thorax_global = sgts_poses["thorax"]
        local_positions["thorax"] = (np.linalg.inv(pelvis_pose) @ thorax_global @ np.array([0, 0, 0, 1]))[:3]
    
     # Torso with respect to pelvis
    if "torso" in sgts_poses:
        torso_global = sgts_poses["torso"]
        thorax_global = sgts_poses["thorax"]
        local_positions["torso"] = (np.linalg.inv(thorax_global) @ torso_global @ np.array([0, 0, 0, 1]))[:3]
        #need to adjust torso frame to aligned it with thorax and pelvis frames.

    # Head with respect to torso
    if "head" in sgts_poses:
        head_global = sgts_poses["head"]
        torso_global = sgts_poses["torso"]
        local_positions["head"] = (np.linalg.inv(thorax_global) @ head_global @ np.array([0, 0, 0, 1]))[:3]

    # Upperarm with respect to torso
    if "upperarmR" in sgts_poses:
        upperarm_global = sgts_poses["upperarmR"]
        torso_global = sgts_poses["torso"]
        local_positions["upperarmR"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    if "upperarmL" in sgts_poses:
        upperarm_global = sgts_poses["upperarmL"]
        torso_global = sgts_poses["torso"]
        local_positions["upperarmL"] = (np.linalg.inv(torso_global) @ upperarm_global @ np.array([0, 0, 0, 1]))[:3]

    # Lowerarm with respect to upperarm
    if "lowerarmR" in sgts_poses:
        lowerarm_global = sgts_poses["lowerarmR"]
        upperarm_global = sgts_poses["upperarmR"]
        local_positions["lowerarmR"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    if "lowerarmL" in sgts_poses:
        lowerarm_global = sgts_poses["lowerarmL"]
        upperarm_global = sgts_poses["upperarmL"]
        local_positions["lowerarmL"] = (np.linalg.inv(upperarm_global) @ lowerarm_global @ np.array([0, 0, 0, 1]))[:3]

    if with_hand:
    # Hand with respect to lowerarm
        if "handR" in sgts_poses:
            hand_global = sgts_poses["handR"]
            lowerarm_global = sgts_poses["lowerarmR"]
            local_positions["handR"] = (np.linalg.inv(lowerarm_global) @ hand_global @ np.array([0, 0, 0, 1]))[:3]

        if "handL" in sgts_poses:
            hand_global = sgts_poses["handL"]
            lowerarm_global = sgts_poses["lowerarmL"]
            local_positions["handL"] = (np.linalg.inv(lowerarm_global) @ hand_global @ np.array([0, 0, 0, 1]))[:3]
            
    # Thigh with respect to pelvis
    if "thighR" in sgts_poses:
        thigh_global = sgts_poses["thighR"]
        local_positions["thighR"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    if "thighL" in sgts_poses:
        thigh_global = sgts_poses["thighL"]
        local_positions["thighL"] = (np.linalg.inv(pelvis_pose) @ thigh_global @ np.array([0, 0, 0, 1]))[:3]

    # Shank with respect to thigh
    if "shankR" in sgts_poses:
        shank_global = sgts_poses["shankR"]
        thigh_global = sgts_poses["thighR"]
        local_positions["shankR"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    if "shankL" in sgts_poses:
        shank_global = sgts_poses["shankL"]
        thigh_global = sgts_poses["thighL"]
        local_positions["shankL"] = (np.linalg.inv(thigh_global) @ shank_global @ np.array([0, 0, 0, 1]))[:3]

    # Foot with respect to shank
    if "footR" in sgts_poses:
        foot_global = sgts_poses["footR"]
        shank_global = sgts_poses["shankR"]
        local_positions["footR"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    
    if "footL" in sgts_poses:
        foot_global = sgts_poses["footL"]
        shank_global = sgts_poses["shankL"]
        local_positions["footL"] = (np.linalg.inv(shank_global) @ foot_global @ np.array([0, 0, 0, 1]))[:3]
    return local_positions

def get_segments_mks_dict(mks_positions)->Dict:
    #This fuction returns a dictionnary containing the segments names, and the corresponding list of lstm
    # mks names attached to the segment
    # Constructing the dictionary to store segment poses
    if 'Head' in mks_positions: #with cosmik set
        sgts_mks_dict = {
        "head": ['Head', 'Nose', 'Right_Ear', 'Left_Ear', 'Right_Eye', 'Left_Eye'],
        "thorax": ['C7'],
        "right_clavicle" : ['RSHO'],
        "left_clavicle" : ['LSHO'],
        
        "upperarmR": ['RMELB', 'RELB'],
        "lowerarmR": ['RWRI', 'RMWRI'],
        "upperarmL" : ['LMELB', 'LELB'],
        "lowerarmL": ['LWRI', 'LMWRI'],
        "pelvis": ['RPSI', 'LPSI', 'RASI', 'LASI'],
        "thighR": ['RKNE', 'RMKNE'],
        "thighL": ['LKNE', 'LMKNE'],
        "shankR": ['RANK', 'RMANK'],
        "shankL": ['LANK', 'LMANK'],
        "footR": ['RHEE' ,'R5MHD','RTOE'],
        "footL": ['LHEE', 'L5MHD', 'LTOE']
    }
    else : #with mocap set
        sgts_mks_dict = {
            "head": ['BHD','RHD','LHD','FHD'],
            "thorax": ['C7','TV8','TV12','SJN','STRN'],
            "right_clavicle" : ['RSHO'],
            "left_clavicle" : ['LSHO'],
            
            "upperarmR": ['RMELB', 'RELB'],
            "lowerarmR": ['RWRI', 'RMWRI'],
            "upperarmL" : ['LMELB', 'LELB'],
            "lowerarmL": ['LWRI', 'LMWRI'],
            "pelvis": ['RPSI', 'LPSI', 'RASI', 'LASI'],
            "thighR": ['RKNE', 'RMKNE'],
            "thighL": ['LKNE', 'LMKNE'],
            "shankR": ['RANK', 'RMANK'],
            "shankL": ['LANK', 'LMANK'],
            "footR": ['RHEE' ,'R5MHD','RTOE'],
            "footL": ['LHEE', 'L5MHD', 'LTOE'],

            "handR": ["RHL2", "RHM5"],
            "handL": ["LHL2", "LHM5"]
        }
    return sgts_mks_dict

def get_local_mks_positions(sgts_poses: Dict, mks_positions: Dict, sgts_mks_dict: Dict, with_hand=True)-> Dict:
    """_Get the local 3D position of the lstms markers_

    Args:
        sgts_poses (Dict): _sgts_poses corresponds to a dictionnary to segments poses and names, constructed from global mks positions_
        mks_positions (Dict): _mks_positions is a dictionnary of lstm mks names and 3x1 global positions_
        sgts_mks_dict (Dict): _sgts_mks_dict a dictionnary containing the segments names, and the corresponding list of lstm mks names attached to the segment_

    Returns:
        Dict: _returns a dictionnary of lstm mks names and their 3x1 local positions_
    """
    mks_local_positions = {}

    for segment, markers in sgts_mks_dict.items():
        # Get the segment's transformation matrix
        segment_pose = sgts_poses[segment]
        
        # Compute the inverse of the segment's transformation matrix
        segment_pose_inv = np.eye(4,4)
        segment_pose_inv[:3,:3] = np.transpose(segment_pose[:3,:3])
        segment_pose_inv[:3,3] = -np.transpose(segment_pose[:3,:3]) @ segment_pose[:3,3]
        for marker in markers:
            if marker in mks_positions:
                # Get the marker's global position
                marker_global_pos = np.append(mks_positions[marker], 1)  # Convert to homogeneous coordinates

                marker_local_pos_hom = segment_pose_inv @ marker_global_pos  # Transform to local coordinates
                marker_local_pos = marker_local_pos_hom[:3]  # Convert back to 3x1 coordinates
                # Store the local position in the dictionary
                mks_local_positions[marker] = marker_local_pos

    return mks_local_positions

def midpoint(p1, p2):
    return 0.5 * (np.array(p1) + np.array(p2))


def compute_hip_joint_center(
    L_ASIS, R_ASIS, L_PSIS, R_PSIS, knee_study, ankle_study, side="right"
):
    """
    Compute hip joint center using Leardini et al. (1999) method.

    """
    ASIS_mid = midpoint(R_ASIS, L_ASIS)
    PSIS_mid = midpoint(R_PSIS, L_PSIS)

    # Distance between ASIS and PSIS centers
    pelvis_depth_vec = ASIS_mid - PSIS_mid
    pelvis_depth = np.linalg.norm(pelvis_depth_vec)

    # Distance between ASIS markers (pelvis width)
    pelvis_width = np.linalg.norm(R_ASIS - L_ASIS)

    ankle_knee_length = np.linalg.norm(ankle_study - knee_study)
    knee_ASIS_length = np.linalg.norm(
        knee_study - (R_ASIS if side == "right" else L_ASIS)
    )
    vertical_adjust = ankle_knee_length + knee_ASIS_length

    hip_y = ASIS_mid[1] - 0.096 * vertical_adjust
    # Compute hip center
    hip_x = ASIS_mid[0] - 0.31 * pelvis_depth
    if side == "right":
        hip_z = ASIS_mid[2] + 0.38 * pelvis_width
    elif side == "left":
        hip_z = ASIS_mid[2] - 0.38 * pelvis_width
    else:
        raise ValueError("Side must be 'right' or 'left'")

    return np.array([hip_x, hip_y, hip_z])


def compute_uptrunk(C7, CLAV):
    vec = CLAV - C7
    norm = np.linalg.norm(vec)
    angle_rad = 8 * np.pi / 180
    return np.array(
        [
            C7[0] + np.cos(angle_rad) * 0.55 * norm,
            C7[1] + np.sin(angle_rad) * 0.55 * norm,
            C7[2],
        ]
    )


def compute_shoulder(SHO, C7, CLAV, side="right"):
    vec = CLAV - C7
    norm = np.linalg.norm(vec)
    angle_rad = 11 * np.pi / 180
    sign = -1 if side == "right" else -1  # both use minus sign in paper

    return np.array(
        [
            SHO[0] + np.cos(angle_rad) * 0.43 * norm,
            SHO[1] + sign * np.sin(angle_rad) * 0.43 * norm,
            SHO[2],
        ]
    )


def compute_joint_centers_from_mks(markers):
    """
    Compute joint center positions and segment lengths from marker positions.

    Parameters
    ----------
    markers : dict[str, np.ndarray]
        Dict of global marker positions. Each value should be shape (3,) or (3,1),
        in either millimeters ("mm") or meters ("m") depending on `units`.
    units : {"mm", "m"}, optional
        Input units for `markers`. Used only for reporting lengths (meters).

    Returns
    -------
    jcp_global : dict[str, np.ndarray]
        Joint centers in GLOBAL frame, each as 1D array shape (3,) in input units.
    segment_lengths : dict[str, float]
        Upper/lower arm segment lengths in meters.
    norms : dict[str, list[float]]
        Elbow inter-epicondyle distances in meters. (Lists so you can append per-frame upstream.)
    """

    # --- helpers ---
    def as_col(x):
        x = np.asarray(x)
        return x.reshape(3, 1) if x.shape != (3, 1) else x

    jcp = {}
    jcp_g = {}

    # Pelvis pose (global)
    pelvis_pose = get_virtual_pelvis_pose(markers)
    pelvis_position = as_col(pelvis_pose[:3, 3])
    pelvis_rotation = pelvis_pose[:3, :3]

    bi_acromial_dist = np.linalg.norm(markers["LSHO"] - markers["RSHO"])
    torso_pose = get_torso_pose(markers)

    # ---- Transform all markers into pelvis (local) frame (do NOT mutate input) ----
    markers_local = {}
    for name, coords in markers.items():
        coords_col = as_col(coords)
        markers_local[name] = transform_to_local_frame(
            coords_col, pelvis_position, pelvis_rotation
        )

    # ---- Shoulders & Neck ----
    try:
        jcp_g["Right_Shoulder"] = markers["RSHO"].reshape(3, 1) + (
            torso_pose[:3, :3].reshape(3, 3)
        ) @ col_vector_3D(0.0, -0.17 * bi_acromial_dist, 0.0)
        jcp_g["Left_Shoulder"] = markers["LSHO"].reshape(3, 1) + (
            torso_pose[:3, :3].reshape(3, 3)
        ) @ col_vector_3D(0.0, -0.17 * bi_acromial_dist, 0.0)

        jcp["Right_Shoulder"] = transform_to_local_frame(
            jcp_g["Right_Shoulder"], pelvis_position, pelvis_rotation
        )
        jcp["Left_Shoulder"] = transform_to_local_frame(
            jcp_g["Left_Shoulder"], pelvis_position, pelvis_rotation
        )
        jcp["Neck"] = compute_uptrunk(markers_local["C7"], markers_local["SJN"])
    except KeyError:
        pass

    # ---- Elbows ----
    try:
        jcp["Right_Elbow"] = midpoint(markers_local["RMELB"], markers_local["RELB"])
        jcp["Left_Elbow"] = midpoint(markers_local["LMELB"], markers_local["LELB"])

    except KeyError:
        pass

    # ---- Wrists ----
    try:
        jcp["Right_Wrist"] = midpoint(markers_local["RMWRI"], markers_local["RWRI"])
        jcp["Left_Wrist"] = midpoint(markers_local["LMWRI"], markers_local["LWRI"])
    except KeyError:
        pass

    # ---- Pelvis & Hips ----
    try:
        R_ASIS = markers_local["RASI"]
        L_ASIS = markers_local["LASI"]
        R_PSIS = markers_local["RPSI"]
        L_PSIS = markers_local["LPSI"]

        jcp["Right_Hip"] = compute_hip_joint_center(
            L_ASIS,
            R_ASIS,
            L_PSIS,
            R_PSIS,
            markers_local["RKNE"],
            markers_local["RANK"],
            side="right",
        )
        jcp["Left_Hip"] = compute_hip_joint_center(
            L_ASIS,
            R_ASIS,
            L_PSIS,
            R_PSIS,
            markers_local["LKNE"],
            markers_local["LANK"],
            side="left",
        )
        jcp["Mid_Hip"] = midpoint(jcp["Right_Hip"], jcp["Left_Hip"])
    except KeyError:
        pass

    # ---- Knees ----
    try:
        jcp["Right_Knee"] = midpoint(markers_local["RMKNE"], markers_local["RKNE"])
        jcp["Left_Knee"] = midpoint(markers_local["LMKNE"], markers_local["LKNE"])
    except KeyError:
        pass

    # ---- Ankles ----
    try:
        jcp["Right_Ankle"] = midpoint(markers_local["RMANK"], markers_local["RANK"])
        jcp["Left_Ankle"] = midpoint(markers_local["LMANK"], markers_local["LANK"])
    except KeyError:
        pass

    # ---- Feet / Toes ----
    try:
        jcp["Right_Heel"] = markers_local["RHEE"]
        jcp["Left_Heel"] = markers_local["LHEE"]
    except KeyError:
        pass

    try:
        jcp["Right_Big_Toe"] = markers_local["RTOE"]
        jcp["Left_Big_Toe"] = markers_local["LTOE"]
    except KeyError:
        pass

    try:
        jcp["Right_Small_Toe"] = markers_local["R5MHD"]
        jcp["Seft_Small_Toe"] = markers_local["L5MHD"]
    except KeyError:
        pass

    # ---- Back to GLOBAL frame ----
    jcp_global = {}
    for name, coords in jcp.items():
        coords_col = as_col(coords)
        # Guard against accidental matrices (e.g., someone returns a 3x3)
        if coords_col.shape != (3, 1):
            # try to coerce; if it fails, skip
            try:
                coords_col = np.asarray(coords).reshape(3, 1)
            except Exception:
                print(
                    f"⚠️ Skipping '{name}' – unexpected shape {np.asarray(coords).shape}"
                )
                continue
        global_coords = transform_to_global_frame(
            coords_col, pelvis_position, pelvis_rotation
        )
        jcp_global[name] = global_coords.flatten()

    return jcp_global


