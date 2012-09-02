#!/usr/bin/python

import getopt, sys, math, cv

import numpy
from numpy import *
import find_circles

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
    return matrix([[1 + (x/z)**2], [x*y/z**2]])

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
    return vstack(sub_jacobian_point(*p, pixel_scale=pixel_scale) for p in points)

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

def matrix_normalize(m):


def matrix_invert(m):
    return vstack([hstack([m[:3, :3].T, -m[:3, 3:4]]), m[3:4, :]])

def solve(world_points, image_points, annotate_image=None):
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

    assert set(world_points.keys()) >= set(image_points.keys())
    keys = list(image_points.keys())
    world_points = hstack([matrix(list(world_points[k]) + [1.0]).T for k in keys])
    image_points = hstack([matrix(image_points[k]).T for k in keys])

    current_mat = matrix_trans(0.0, 0.0, 500.0) 
    current_ps  = 500.0

    def camera_to_image(m, ps):
        return ps * matrix([[c[0, 0] / c[0, 2], c[0, 1] / c[0, 2]] for c in m.T]).T

    def test_jacobian(J, camera_points, ps):
        theta = 0.001
        
        params = [("Trans X", matrix_trans(theta, 0.0, 0.0)),
                  ("Trans Y", matrix_trans(0.0, theta, 0.0)),
                  ("Trans Z", matrix_trans(0.0, 0.0, theta)),
                  ("Rot X", matrix_rotate_x(theta)),
                  ("Rot Y", matrix_rotate_y(theta)),
                  ("Rot Z", matrix_rotate_z(theta))]

        params = [(name, camera_to_image(mat * camera_points, ps)) for name, mat in params]
        params += [("Zoom", camera_to_image(camera_points, ps + theta))]

        points_before = camera_to_image(camera_points, ps)
        for idx, (name, points_after) in enumerate(params):
            points_delta = J * matrix([[0.0]] * idx + [[theta]] + [[0.0]] * (6 - idx))

            print "Testing %s" % name
            print "Estimate: %s" % points_delta.flatten()
            print "Actual: %s" % (points_after - points_before).T
            print "Diff: %s" % (points_delta.flatten() - (points_after - points_before).T.flatten())
            print


    while True:
        camera_points = current_mat * world_points
        err = image_points - camera_to_image(camera_points, current_ps)
        #print "Error: %s" % err

        J = make_jacobian(camera_points.T[:, :3], current_ps)
        
        #test_jacobian(J, camera_points, current_ps)

        err = err.T.reshape(2 * len(keys), 1)
        param_delta = numpy.linalg.pinv(J) * (0.01 * err)

        print "Error: %f" % (err.T * err)[0, 0]

        print "Param delta: %s" % param_delta
        current_mat = matrix_trans(param_delta[0, 0],
                                   param_delta[1, 0],
                                   param_delta[2, 0]) * current_mat
        current_mat = matrix_rotate_x(param_delta[3, 0]) * current_mat
        current_mat = matrix_rotate_y(param_delta[4, 0]) * current_mat
        current_mat = matrix_rotate_z(param_delta[5, 0]) * current_mat
        current_ps += param_delta[6, 0]

        import time
        time.sleep(1)
    
    return matrix_invert(current_mat), current_ps
    

if __name__ == "__main__":
    world_circles = dict(("%d%s" % (x + 8 * y, l), (25.0 * x, 85.0 * y + (0 if l == 'a' else 50.0), 0.0)) for y in range(3) for x in range(8) for l in ('a', 'b'))

    optlist, args = getopt.getopt(sys.argv[1:], 'i:')

    for opt, param in optlist:
        if opt == "-i":
            in_file_name = param

    if not in_file_name:
        raise Exception("Usage: %s -i <input image>" % sys.argv[0])

    print "Loading image (black and white)"
    image = cv.LoadImage(in_file_name, False)
    print "Loading image (colour)"
    color_image = cv.LoadImage(in_file_name, True)
    print "Finding labelled circles"
    image_circles = find_circles.find_labelled_circles(image,
                                                       annotate_image=color_image,
                                                       centre_origin=True)
    print image_circles
    print "Solving"
    solve(world_circles, image_circles)

