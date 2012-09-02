from numpy import *
import math

__all__ = ['solve']

# sub_jacobian_point_<transformation>()
#
# Gives:
#    [[dPx/dtheta], [dPy/dtheta]]
#
# where Px,Py are the point projection functions:
#    Px(x, y, z) = x / z
#    Py(x, y, z) = y / z
# 
# and theta is the parameter of <transformation>.
# Transformations are applied to the camera, and rotations are CCW about the
# axis of rotation.

def sub_jacobian_point_translation_x(x, y, z):
    return matrix([[-1/z], [0.0]])

def sub_jacobian_point_translation_y(x, y, z):
    return matrix([[0.0], [-1/z]])

def sub_jacobian_point_translation_z(x, y, z):
    return matrix([[x/z**2], [y/z**2]])

def sub_jacobian_point_rotation_x(x, y, z):
    return matrix([[x*y/z**2], [1 + (y/z)**2]])

def sub_jacobian_point_rotation_y(x, y, z):
    return matrix([[1 + (x/z)**2], [x*y/z**2]])

def sub_jacobian_point_rotation_z(x, y, z):
    return matrix([[y/z], [-x/z]])

def sub_jacobian_point(x, y, z, pixel_scale):
    """
    Return the Jacobian for parameters translation in x, y and z, rotation in
    x, y and z, and zoom.
    """
    fns = [sub_jacobian_point_translation_x,
           sub_jacobian_point_translation_y,
           sub_jacobian_point_translation_z,
           sub_jacobian_point_rotation_x,
           sub_jacobian_point_rotation_y,
           sub_jacobian_point_rotation_z]

    out = hstack([pixel_scale * fn(x, y, z) for fn in fns])
    out = hstack([out, matrix([[x/z], [y/z]])])

    return out

def make_jacobian(points):
    return vstack(sub_jacobian_point(*p) for p in points)

def matrix_rotate_x(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return matrix([[1.0, 0.0, 0.0, 0.0],
                   [0.0,   c,  -s, 0.0],
                   [0.0,   s,   c, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

def matrix_rotate_y(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return matrix([[  c, 0.0,   s, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [ -s, 0.0,   c, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

def matrix_rotate_z(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return matrix([[ c,   -s, 0.0, 0.0],
                   [ s,    c, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 1.0]])

def matrix_trans(x, y, z):
    return matrix([[1.0, 0.0, 0.0,   x],
                   [0.0, 1.0, 0.0,   y],
                   [0.0, 0.0, 1.0,   z],
                   [0.0, 0.0, 0.0, 1.0]])

def solve(world_points, camera_points)
    """
    Find a camera's orientation and pixel scale given a set of world
    coordinates and corresponding set of camera coordinates.

    world_points: Dict mapping point names to triples corresponding with world
                  x, y, z coordinates.
    camera_points: Dict mapping point names to triples corresponding with
                  camera x, y coordinates. Coordinates are translated such that
                  0, 0 corresponds with the centre of the image.

    Return: 4x4 matrix representing the camera's orientation, and a pixel
            pixel scale.
    """

    assert world_points.keys() == camera_points.keys()

    keys = list(world_points.keys())

    current_mat = matrix_trans(0.0, 0.0, -500.0) 
    current_ps  = 500.0


if __name__ == "__main__":
    circles = dict(("%d%s" % (x + 8 * y, l, 0.0), (25.0 * x, 85.0 * y + (0 if l == 'a' else 50.0))) for y in range(3) for x in range(8) for l in ('a', 'b'))

