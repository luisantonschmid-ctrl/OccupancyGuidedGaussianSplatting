import numpy as np

def estimate_transformation(source_points, target_points):
    """
    Estimate a rigid 3D transformation from corresponding points.

    Parameters
    ----------
    source_points : (N, 3) ndarray
        Points in the initial position.
    target_points : (N, 3) ndarray
        Corresponding points in the target position.

    Returns
    -------
    T : (4, 4) ndarray
        Homogeneous transformation matrix.
    """
    source_points = np.asarray(source_points, dtype=float)
    target_points = np.asarray(target_points, dtype=float)

    assert source_points.shape == target_points.shape
    assert source_points.ndim == 2 and source_points.shape[1] == 3
    assert len(source_points) >= 3

    # Calculate centroids
    centroid_source = source_points.mean(axis=0)
    centroid_target = target_points.mean(axis=0)

    # Center the points
    P = source_points - centroid_source
    Q = target_points - centroid_target

    # Compute covariance matrix
    H = P.T @ Q

    # Singular value decomposition
    U, S, Vt = np.linalg.svd(H)

    # Calculate rotation
    R = Vt.T @ U.T

    # Ensure a proper rotation (det(R) = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Calculate translation
    t = centroid_target - R @ centroid_source

    # Construct homogeneous transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

source_points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

target_points = np.array([
    [2, 3, 1],
    [2, 4, 1],
    [1, 3, 1],
    [2, 3, 2]
])

T = estimate_transformation(source_points, target_points)

print(T)

points_transformed = source_points @ T[:3, :3].T + T[:3, 3]

print(points_transformed)