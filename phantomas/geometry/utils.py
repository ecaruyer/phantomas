"""
This module contains utility functions for geometry.
"""
import numpy as np


def rotation_matrix(u, v):
    """Given two vectors :math:`\\mathbf{u}` and :math:`\\mathbf{v}`, 
    computes a rotation matrix, :math:`\\mathbf{R}`, s.t. 
    :math:`\\mathbf{R}\\cdot \\mathbf{u} = \\mathbf{v}`.

    Parameters
    ----------
    u : array-like, shape (3, )
    v : array-like, shape (3, )

    Returns
    -------
    R : array-like, shape (3, 3)

    """
    # the axis is given by the product u x v
    u = u / np.sqrt((u ** 2).sum())
    v = v / np.sqrt((v ** 2).sum())
    w = np.array([u[1] * v[2] - u[2] * v[1], \
        u[2] * v[0] - u[0] * v[2], \
        u[0] * v[1] - u[1] * v[0]])
    if (w ** 2).sum() < 1e-9:
        #The vectors u and v are collinear
        return np.eye(3)
    # computes sine and cosine
    c = np.dot(u, v)
    s = np.sqrt((w ** 2).sum())
    # Q = [[][][]]
    # R = w.w^T + c.(I - w.w^T) + s.Q
    w = w / s
    P = np.outer(w, w)
    Q = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = P + c * (np.eye(3) - P) + s * Q
    return R

