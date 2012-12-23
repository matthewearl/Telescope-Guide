#!/usr/bin/python

import getopt, sys, math, cv

import gen_target
import numpy
from numpy import *
import find_circles
import util

__all__ = ['solve']

STEP_FRACTION = 0.1
ERROR_CUTOFF = 0.0001

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

def calculate_barrel_distortion(bd, sx, sy):
    """
    Calculate distorted point (px, py) given undistorted point (sy,sy), and
    distortion parameter bd.

    Solve by inverting ru = rd * (1 + bd * rd**2) with Newton's method.
    """

    ru = math.sqrt(sx**2 + sy**2)

    x = ru

    for i in xrange(20):
        x = x - (x * (1. + bd * x**2) - ru) / (1. + 3 * bd * x**2)
    rd = x

    px = sx * rd / ru
    py = sy * rd / ru

    return px, py

def make_barrel_distortion_jacobian(bd, sx, sy):
    """
    Return the jacobian for the barrel distortion function, parameterized by:
        - bd: Amount of barrel distortion.
        - sx: Pixel x-position after pixel scaling.
        - sy: Pixel y-position after pixel scaling.

    Barrel distortion shifts points radially outwards according to the following
    function:

    ru = rd * (1 + bd * rd**2)

    Where ru/rd are the undistorted/distorted distance from the image centre.

    More specifically, this function returns:

    | dpx/dsx dpx/dsy dpx/dbd |
    | dpy/dsx dpy/dsy dpy/dbd |

    Where px,py are the barrel distorted points, sx,sy are the unbarrel distorted
    points, and bd is the parameter in the above barrel distortion equation.
    """

    px, py = calculate_barrel_distortion(bd, sx, sy)

    # Calculate the following derivatives immediately from the barrel distortion
    # equations in terms of sx, sy, px and py:
    #  sx = px * (1 + bd * (px**2 + py**2))
    #  sy = py * (1 + bd * (px**2 + py**2))

    dsx_by_dpx = (1. + 3. * bd * px**2 + bd * py**2)
    dsy_by_dpx = 2. * bd * px * py
    dsx_by_dpy = 2. * bd * px * py
    dsy_by_dpy = (1. + 3. * bd * py**2 + bd * px**2)
    dsx_by_dbd = px * (px**2 + py**2)
    dsy_by_dbd = py * (px**2 + py**2)

    # From the chain rule obtain:
    #
    #  1 = dpx/dpx = dpx/dsx * dsx/dpx + dpx/dsy * dsy/dpx
    #  0 = dpx/dpy = dpx/dsx * dsx/dpy + dpx/dsy * dsy/dpy
    #  0 = dpy/dpx = dpy/dsx * dsx/dpx + dpy/dsy * dsy/dpx
    #  1 = dpy/dpy = dpy/dsx * dsx/dpy + dpy/dsy * dsy/dpy
    #
    #  Equivalently:
    #
    #   I = [ dpx/dsx dpx/dsy ] [ dsx/dpx dsx/dpy ]
    #       [ dpy/dsx dpy/dsy ] [ dsy/dpx dsy/dpy ] 
    #
    #  (RHS is assigned to s_jacobian, LHS is assigned to p_jacobian)

    s_jacobian = matrix([[dsx_by_dpx, dsx_by_dpy],
                         [dsy_by_dpx, dsy_by_dpy]])
    p_jacobian = s_jacobian.I

    # Obtain dpx/dbd and dpy/dbd using the implicit function theorem
    # for two variables.

    dpx_by_dbd = -dsx_by_dbd / dsx_by_dpx
    dpy_by_dbd = -dsy_by_dbd / dsy_by_dpy

    return hstack([p_jacobian, matrix([[dpx_by_dbd],[dpy_by_dbd]])])

def sub_jacobian_point(x, y, z, pixel_scale, bd):
    """
    Return the Jacobian for parameters translation in x, y and z, rotation in
    x, y and z, and zoom. I.e.

    | dpx/dtx dpx/dty dpx/dtz dpx/drx dpx/dry dpx/drz dpx/dps dpx/dbd |
    | dpy/dtx dpy/dty dpy/dtz dpy/drx dpy/dry dpy/drz dpy/dps dpy/dbd |

    Where:
        p[xy] are the pixel positions after pixel scaling and barrel distortion.
        [rt][xyz] are parameters for rotations/translations about the axes.
        ps is the pixel scaling parameter.
        bd is the barrel distortion parameter.
    """
    fns = [sub_jacobian_point_translation_x,
           sub_jacobian_point_translation_y,
           sub_jacobian_point_translation_z,
           sub_jacobian_point_rotation_x,
           sub_jacobian_point_rotation_y,
           sub_jacobian_point_rotation_z]

    # First work out the jacobian without barrel distortion
    out = hstack([pixel_scale * fn(x, y, z) for fn in fns])
    out = hstack([out, matrix([[x/z], [y/z]])])

    # Add on an extra row for dbd/d{param}
    out = vstack([out, matrix([[0.] * out.shape[1]])])

    # Add on an extra column for d{sx,sy,bd}/dbd
    out = hstack([out, matrix([[0.], [0.], [1.]])])

    # Compose with the barrel-distortion jacobian
    sx = pixel_scale * x / z
    sy = pixel_scale * y / z
    out = make_barrel_distortion_jacobian(bd, sx, sy) * out

    return out

def make_single_jacobian(points, pixel_scale, bd):
    points = array(points)
    return vstack(sub_jacobian_point(*p, pixel_scale=pixel_scale, bd=bd) for p in points)

def make_jacobian(points, keys, pixel_scale, bd):
    """
    Make the full jacobian matrix. The 6*len(keys) + 2 columns correspond with:
      - 6 parameters for each image:
        - Camera x, y, z translation
        - Camera x, y, z rotation
      - Pixel scale
      - Barrel distortion

    The 2 * len(points) rows correspond with the image coordinates of each point
    in each image.
    """

    Js = []
    idx = 0
    keys_idx = 0
    while idx < len(points) and keys_idx < len(keys):
        J = make_single_jacobian(points[idx:(idx + len(keys[keys_idx])), :], pixel_scale, bd)
        M = zeros((J.shape[0], 6*len(keys) + 2))
        M[:, (6*keys_idx):(6*keys_idx + 6)] = J[:, :6]
        M[:, -2:] = J[:, 6:]
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

def solve(world_points_in, image_points, annotate_images=None,
          initial_matrices=None, initial_ps=2500., change_ps=False, change_bd=False):
    """
    Find a camera's orientation and pixel scale given a set of world
    coordinates and corresponding set of camera coordinates.

    world_points: Dict mapping point names to triples corresponding with world
                  x, y, z coordinates.
    image_points: Array of dicts mapping point names to triples corresponding
                  with camera x, y coordinates. Coordinates are translated such
                  that 0, 0 corresponds with the centre of the image.
                  One array element per source image.
    annotate_images: Optional array of images to annotate with the fitted
                     points.
    initial_matrices: Optional set of initial rotation matrices.
    change_ps: If True, allow the pixel scale (zoom) to be varied. Algorithm
               can be unstable if initial guess is inaccurate.
    change_bd: If True, allow the barrel distortion to be varied. Algorithm
               can be unstable if initial guess is inaccurate.

    Return: 4x4 matrix representing the camera's orientation, and a pixel
            pixel scale.
    """

    assert all(set(world_points_in.keys()) >= set(p.keys()) for p in image_points)
    keys = [list(p.keys()) for p in image_points]
    world_points = [hstack([matrix(list(world_points_in[k]) + [1.0]).T for k in sub_keys]) for sub_keys in keys]
    image_points = hstack([hstack([matrix(p[k].get_centre()).T for k in sub_keys]) for p, sub_keys in zip(image_points, keys)])

    print image_points

    if initial_matrices:
        current_mat = [matrix_invert(m) for m in initial_matrices]
    else:
        current_mat = [matrix_trans(0.0, 0.0, 500.0)] * len(keys)
    current_ps = initial_ps
    current_bd = 0.0

    def camera_to_image(m, ps, bd):
        def map_point(c):
            px, py = calculate_barrel_distortion(bd, ps * c[0, 0] / c[0, 2], ps * c[0, 1] / c[0, 2])
            return [px, py]
        return matrix([map_point(c) for c in m.T]).T

    last_err_float = None
    while True:
        # Calculate the Jacobian
        camera_points = hstack([m * p for m, p in zip(current_mat, world_points)])
        err = image_points - camera_to_image(camera_points, current_ps, current_bd)
        J = make_jacobian(camera_points.T[:, :3], keys, current_ps, current_bd)
        if not change_ps:
            J = hstack([J[:, :-2], J[:, -1:]])
        if not change_bd:
            J = J[:, :-1]

        # Invert the Jacobian and calculate the change in parameters.
        # Limit angle changes to avoid chaotic behaviour.
        err = err.T.reshape(2 * sum(len(sub_keys) for sub_keys in keys), 1)
        param_delta = numpy.linalg.pinv(J) * (STEP_FRACTION * err)

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

        if change_bd:
            current_bd += param_delta[6 * len(keys) + 1, 0]

    if annotate_images:
        all_keys = list(world_points_in.keys())
        all_world_points = hstack([matrix(list(world_points_in[k]) + [1.0]).T for k in all_keys])
        for i, annotate_image in enumerate(annotate_images):
            all_camera_points = current_mat[i] * all_world_points
            util.draw_points(annotate_image,
                             dict(zip(all_keys, camera_to_image(all_camera_points, current_ps, current_bd).T)))
    
    return [matrix_invert(M) for M in current_mat], current_ps, current_bd
    
import random

def test_calculate_barrel_distortion():
    sx = 2000 * random.random() - 1000.0
    sy = 2000 * random.random() - 1000.0
    bd = 2. * random.random()

    px, py = calculate_barrel_distortion(bd, sx, sy)

    ru = math.sqrt(sx**2 + sy**2)
    rd = math.sqrt(px**2 + py**2)

    print "%f %f" % (ru, rd * (1. + bd * rd**2))


def test_barrel_distortion_jacobian():
    sx = 2000 * random.random() - 1000.0
    sy = 2000 * random.random() - 1000.0
    bd = .000001 * random.random()

    dsx = 0.02 * random.random() - 0.01
    dsy = 0.02 * random.random() - 0.01
    dbd = -0.00000002 * random.random()

    px1, py1 = calculate_barrel_distortion(bd, sx, sy)
    px2, py2 = calculate_barrel_distortion(bd + dbd, sx + dsx, sy + dsy)

    print "---"
    print matrix([[px2 - px1, py2 - py1]])
    
    J = make_barrel_distortion_jacobian(bd, sx, sy)
    print (J * matrix([[dsx, dsy, dbd]]).T).T


if __name__ == "__main__":
    world_circles = gen_target.get_targets()

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
                                                        find_ellipses=True)
                        for image, color_image in zip(images, color_images)]
    print image_circles
    print "Solving"
    Ms, ps, bd = solve(world_circles, image_circles, annotate_images=color_images)

    print "Solving for zoom"
    Ms, ps, bd = solve(world_circles, image_circles, annotate_images=color_images, initial_matrices=Ms, change_ps=True)

    print "Solving for barrel distortion"
    print solve(world_circles, image_circles, annotate_images=color_images, initial_matrices=Ms, initial_ps=ps, change_ps=True, change_bd=True)

    for i, out_file_name in enumerate(out_file_names):
        cv.SaveImage(out_file_name, color_images[i])

