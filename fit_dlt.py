from numpy import *
import util

# Implementation of algorithm described here:
#
# http://users.cecs.anu.edu.au/~hartley/Papers/CVPR99-tutorial/tut_4up.pdf
#
# Compute the 3x4 camera matrix using a direct linear translation

def correspondence_matrix(world_point, image_point):
    world_point = list(world_point) + [1.]
    return matrix([world_point + [0.] * 4 + [-image_point[0]*x for x in world_point],
                   [0.] * 4 + world_point + [-image_point[1]*x for x in world_point]])

def solve(world_points, image_points, annotate_image=None):
    """
    Find a camera's orientation and intrinsic parameters given a set of world
    coordinates and corresponding set of camera coordinates.

    world_points: Dict mapping point names to triples corresponding with world
                  x, y, z coordinates.
    image_points: Dict mapping point names to triples corresponding with
                  camera x, y coordinates. Coordinates are translated such that
                  0, 0 corresponds with the centre of the image.

    Return: 4x4 matrix representing the camera's orientation, and a pixel
            pixel scale.
    """

    assert set(world_points.keys()) >= set(image_points.keys())
    keys = list(image_points.keys())
    # Need at least 6 points (12 DOF) to solve the 3x4 matrix
    assert len(keys) >= 6

    M = vstack([correspondence_matrix(world_points[key], image_points[key]) for key in keys])

    eig_vals, eig_vecs = np.linalg.eig(M)
    P = eig_vecs[eig_vals.argsort()][0]
    P = P.reshape((3, 4))

    if annotate_image:


