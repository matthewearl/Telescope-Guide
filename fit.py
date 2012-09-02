#!/usr/bin/python

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
# Transformations are applied to the world, and rotations are CCW about the
# axis of rotation.

def sub_jacobian_point_translation_x(x, y, z):
    return matrix([[1/z], [0.0]])

def sub_jacobian_point_translation_y(x, y, z):
    return matrix([[0.0], [1/z]])

def sub_jacobian_point_translation_z(x, y, z):
    return matrix([[-x/z**2], [-y/z**2]])

def sub_jacobian_point_rotation_x(x, y, z):
    return matrix([[-x*y/z**2], [-1 - (y/z)**2]])

def sub_jacobian_point_rotation_y(x, y, z):
    return matrix([[-1 - (x/z)**2], [-x*y/z**2]])

def sub_jacobian_point_rotation_z(x, y, z):
    return matrix([[-y/z], [x/z]])

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

def make_jacobian(points, pixel_scale):
    points = array(points)
    return vstack(sub_jacobian_point(*p, pixel_scale) for p in points)

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

def matrix_invert(m):
    return vstack([hstack([m[:3, :3].T, -m[:3, 3:4]]), m[3:4, :]])

def solve(world_points, image_points):
    """
    Find a camera's orientation and pixel scale given a set of world
    coordinates and corresponding set of camera coordinates.

    world_points: Dict mapping point names to triples corresponding with world
                  x, y, z coordinates.
    image_points: Dict mapping point names to triples corresponding with
                  camera x, y coordinates. Coordinates are translated such that
                  0, 0 corresponds with the centre of the image.

    Return: 4x4 matrix representing the camera's orientation, and a pixel
            pixel scale.
    """

    assert world_points.keys() >= image_points.keys()
    keys = list(image_points.keys())
    world_points = hstack([matrix(list(world_points[k]) + [1.0]).T for k in keys])
    image_points = hstack([matrix(image_points[k]).T for k in keys])

    current_mat = matrix_trans(0.0, 0.0, 500.0) 
    current_ps  = 500.0

    def camera_to_image(m, ps):
        return matrix([[c[0] / c[2], c[1] / c[2]] for c in m.T]).T * ps

    while True:
        camera_points = current_mat * world_points
        err = image_points - camera_to_screen(camera_points, current_ps)
        print "Error: %s" % err

        J = make_jacobian(camera_points.T, current_ps)

        param_delta = ((J.T . J).I * J.T) * (0.1 * err)

        current_mat = matrix_rotate_x(param_delta[0]) * current_mat
        current_mat = matrix_rotate_y(param_delta[1]) * current_mat
        current_mat = matrix_rotate_z(param_delta[2]) * current_mat
        current_mat = matrix_trans(*param_delta[3:6]) * current_mat

        current_ps += param_delta[6]
    
    return matrix_invert(current_mat), current_ps
    

if __name__ == "__main__":
    world_circles = dict(("%d%s" % (x + 8 * y, l), (25.0 * x, 85.0 * y + (0 if l == 'a' else 50.0), 0.0)) for y in range(3) for x in range(8) for l in ('a', 'b'))

    optlist, args = getopt.getopt(sys.argv[1:], 'i:')

    for opt, param in optlist:
        if opt == "-i":
            in_file_name = param

    if not in_file_name or not out_file_name:
        raise Exception("Usage: %s -i <input image>" % sys.argv[0])

    image = cv.LoadImage(in_file_name, False)
    color_image = cv.LoadImage(in_file_name, True)
    image_circles = find_labelled_circles(image, thresh_file_name, color_image)

    solve(world_circles, image_circles)

