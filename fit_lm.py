#!/usr/bin/python

import getopt, sys, math, cv

import numpy
from numpy import *
import find_circles
import util

__all__ = ['solve']

MAX_ANGLE_DELTA = 30.0 * math.pi / 180
MAX_DELTA = 5000.0
STEP_FRACTION = 0.01
ERROR_CUTOFF = 0.000001

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

def make_single_jacobian(points, pixel_scale):
    points = array(points)
    return vstack(sub_jacobian_point(*p, pixel_scale=pixel_scale) for p in points)

def make_jacobian(points, keys, pixel_scale):
    Js = []
    idx = 0
    keys_idx = 0
    while idx < len(points) and keys_idx < len(keys):
        J = make_single_jacobian(points[idx:(idx + len(keys[keys_idx])), :], pixel_scale)
        M = zeros((J.shape[0], 6*len(keys) + 1))
        M[:, (6*keys_idx):(6*keys_idx + 6)] = J[:, :6]
        M[:, -1:] = J[:, 6:]
        Js.append(M)

        idx += len(keys[keys_idx])
        keys_idx += 1

    assert idx == len(points) and keys_idx == len(keys)

    return vstack(Js)


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


def matrix_normalize(m):
    m[2:3, :3] = cross(m[0:1, :3], m[1:2, :3])
    m[2:3, :3] /= float(matrix(m[2:3, :3]) * matrix(m[2:3, :3]).T)
    m[0:1, :3] = cross(m[1:2, :3], m[2:3, :3])
    m[0:1, :3] /= float(matrix(m[0:1, :3]) * matrix(m[0:1, :3]).T)
    m[1:2, :3] = cross(m[2:3, :3], m[0:1, :3])
    m[1:2, :3] /= float(matrix(m[1:2, :3]) * matrix(m[1:2, :3]).T)

    m[3:4, :] = matrix([[0.0, 0.0, 0.0, 1.0]])

def solve(world_points_in, image_points, annotate_images=None, initial_matrices=None, change_ps=False):
    """
    Find a camera's orientation and pixel scale given a set of world
    coordinates and corresponding set of camera coordinates.

    world_points: Dict mapping point names to triples corresponding with world
                  x, y, z coordinates.
    image_points: Array of dicts mapping point names to triples corresponding with
                  camera x, y coordinates. Coordinates are translated such that
                  0, 0 corresponds with the centre of the image.
                  One array element per source image.

    Return: 4x4 matrix representing the camera's orientation, and a pixel
            pixel scale.
    """

    assert all(set(world_points_in.keys()) >= set(p.keys()) for p in image_points)
    keys = [list(p.keys()) for p in image_points]
    world_points = [hstack([matrix(list(world_points_in[k]) + [1.0]).T for k in sub_keys]) for sub_keys in keys]
    image_points = hstack([hstack([matrix(p[k]).T for k in sub_keys]) for p, sub_keys in zip(image_points, keys)])

    if initial_matrices:
        current_mat = [matrix_invert(m) for m in initial_matrices]
    else:
        current_mat = [matrix_trans(0.0, 0.0, 500.0)] * len(keys)
    current_ps  = 3059.7776822502801

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

    last_err_float = None
    while True:
        # Calculate the Jacobian
        camera_points = hstack([m * p for m, p in zip(current_mat, world_points)])
        err = image_points - camera_to_image(camera_points, current_ps)
        J = make_jacobian(camera_points.T[:, :3], keys, current_ps)
        if not change_ps:
            J = J[:, :-1]
        #test_jacobian(J, camera_points, current_ps)

        # Invert the Jacobian and calculate the change in parameters.
        # Limit angle changes to avoid chaotic behaviour.
        err = err.T.reshape(2 * sum(len(sub_keys) for sub_keys in keys), 1)
        param_delta = numpy.linalg.pinv(J) * (STEP_FRACTION * err)
        #max_angle = reduce(max, list(array(param_delta)[3:6].flatten()))
        #if max_angle > MAX_ANGLE_DELTA:
        #    param_delta *= (MAX_ANGLE_DELTA / max_angle)
        #max_delta = reduce(max, list(array(param_delta).flatten()))
        #if max_delta > MAX_DELTA:
        #    param_delta = (MAX_DELTA / max_delta) * param_delta

        # Calculate the error (as sum of squares), and abort if the error has
        # stopped decreasing.
        err_float = (err.T * err)[0, 0]
        print "Error: %f" % err_float
        if last_err_float != None and abs(err_float - last_err_float) < ERROR_CUTOFF:
            break
        last_err_float = err_float

        # Apply the parameter delta.
        for i in xrange(len(keys)):
            current_mat[i] = matrix_trans(param_delta[6 * i + 0, 0], param_delta[6 * i + 1, 0], param_delta[6 * i + 2, 0]) * current_mat[i]
            current_mat[i] = matrix_rotate_x(param_delta[6 * i + 3, 0]) * current_mat[i]
            current_mat[i] = matrix_rotate_y(param_delta[6 * i + 4, 0]) * current_mat[i]
            current_mat[i] = matrix_rotate_z(param_delta[6 * i + 5, 0]) * current_mat[i]
            matrix_normalize(current_mat[i])

        if change_ps:
            current_ps += param_delta[6 * len(keys), 0]

    if annotate_images:
        all_keys = list(world_points_in.keys())
        all_world_points = hstack([matrix(list(world_points_in[k]) + [1.0]).T for k in all_keys])
        for i, annotate_image in enumerate(annotate_images):
            all_camera_points = current_mat[i] * all_world_points
            util.draw_points(annotate_image,
                             dict(zip(all_keys, camera_to_image(all_camera_points, current_ps).T)))
    
    return [matrix_invert(M) for M in current_mat], current_ps
    

if __name__ == "__main__":
    world_circles = util.get_circle_pattern(roll_radius=71.)

    optlist, args = getopt.getopt(sys.argv[1:], 'i:o:')

    in_file_names = []
    out_file_names = []
    for opt, param in optlist:
        if opt == "-i":
            in_file_names += [param]
        if opt == "-o":
            out_file_names += [param]

    if not in_file_names:
        raise Exception("Usage: %s -i <input image> [-o <output image>]" % sys.argv[0])

    print "Loading images (black and white)"
    images = [cv.LoadImage(in_file_name, False) for in_file_name in in_file_names]
    print "Loading image (colour)"
    color_images = [cv.LoadImage(in_file_name, True) for in_file_name in in_file_names]
    print "Finding labelled circles"
    image_circles = [find_circles.find_labelled_circles(image,
                                                        annotate_image=color_image,
                                                        centre_origin=True)
                        for image, color_image in zip(images, color_images)]
    print image_circles
    print "Solving"
    Ms, ps = solve(world_circles, image_circles, annotate_images=color_images)

    print "Solving for zoom"
    print solve(world_circles, image_circles, annotate_images=color_images, initial_matrices=Ms, change_ps=True)

    for i, out_file_name in enumerate(out_file_names):
        cv.SaveImage(out_file_name, color_images[i])

