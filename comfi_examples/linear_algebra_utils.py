import numpy as np
from scipy import signal


def check_orthogonality(matrix: np.ndarray):
    # Vecteurs colonnes
    X = matrix[:3, 0]
    Y = matrix[:3, 1]
    Z = matrix[:3, 2]
    
    # Calcul des produits scalaires
    dot_XY = np.dot(X, Y)
    dot_XZ = np.dot(X, Z)
    dot_YZ = np.dot(Y, Z)
    
    # Tolérance pour les erreurs numériques
    tolerance = 1e-6
    
    print(f"Dot product X.Y: {dot_XY}")
    print(f"Dot product X.Z: {dot_XZ}")
    print(f"Dot product Y.Z: {dot_YZ}")
    
    assert np.abs(dot_XY) < tolerance, "Vectors X and Y are not orthogonal"
    assert np.abs(dot_XZ) < tolerance, "Vectors X and Z are not orthogonal"
    assert np.abs(dot_YZ) < tolerance, "Vectors Y and Z are not orthogonal"
    
def orthogonalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Function that takes as input a matrix and orthogonalizes it
    Its mainly used to orthogonalize rotation matrices constructed by hand
    """
    # Perform Singular Value Decomposition
    U, _, Vt = np.linalg.svd(matrix)
    # Reconstruct the orthogonal matrix
    orthogonal_matrix = U @ Vt
    # Ensure the determinant is 1
    if np.linalg.det(orthogonal_matrix) < 0:
        U[:, -1] *= -1
        orthogonal_matrix = U @ Vt
    return orthogonal_matrix


def col_vector_3D(a, b, c):
    return np.array([[float(a)], [float(b)], [float(c)]], dtype=np.float64)


def transform_to_local_frame(D, origin, rotation_matrix):
    # Compute D relative to B
    D_relative = D - origin

    # Transform D to the local frame
    D_local = rotation_matrix.T @ D_relative

    return D_local


def transform_to_global_frame(D, origin, rotation_matrix):
    D_global = rotation_matrix @ D + origin
    return D_global


def butterworth_filter(data, cutoff_frequency, order=5, sampling_frequency=60):
    nyquist = 0.5 * sampling_frequency
    if not 0 < cutoff_frequency < nyquist:
        raise ValueError("Cutoff frequency must be between 0 and Nyquist frequency.")
    b, a = signal.butter(order, cutoff_frequency / nyquist, btype="low", analog=False)
    return signal.filtfilt(b, a, data, axis=0)
