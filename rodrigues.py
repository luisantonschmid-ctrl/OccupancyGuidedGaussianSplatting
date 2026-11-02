import numpy as np


def rotation_matrix_from_vectors(vec_a, vec_b):
  """Find the rotation matrix that aligns vec_a to vec_b."""
  a = vec_a / np.linalg.norm(vec_a)
  b = vec_b / np.linalg.norm(vec_b)
  v = np.cross(a, b)
  c = np.dot(a, b)
  s = np.linalg.norm(v)

  if s == 0:
    return np.identity(3)  # Vectors are identical or opposite

  # Skew-symmetric cross-product matrix of v
  v_x = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

  # Rodrigues' rotation formula
  r = np.identity(3) + v_x + (v_x @ v_x) * ((1 - c) / (s**2))
  return r


# Example usage:
vec_a = np.array([12.27192956,-2.35717546,13.99321994])
vec_b = np.array([2.7520804,-1.2885737,18.513031])
R = rotation_matrix_from_vectors(vec_a, vec_b)
print(R)