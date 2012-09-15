#!/usr/bin/python

import getopt, sys, cv
import numpy
import scipy.linalg
from numpy import *
import find_circles
import util

__all__ = ['solve']

# Implementation of algorithm described here:
#
# http://users.cecs.anu.edu.au/~hartley/Papers/CVPR99-tutorial/tut_4up.pdf
#
# Compute the 3x4 camera matrix using a direct linear transformation

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
    eig_vals, eig_vecs = numpy.linalg.eig(M.T * M)
    P = (eig_vecs.T[eig_vals.argmin()]).T
    
    P = P.reshape((3, 4))
    K, R = map(matrix, scipy.linalg.rq(P[:, :3]))
    t = K.I * P[:, 3:]
    R = hstack((R, t))

    K = K / K[2,2]

    #K[0, 1] = 0.0
    #K[0, 2] = 0.0
    #K[1, 2] = 0.0

    P = K * R

    if annotate_image:
        all_keys = list(world_points.keys())
        world_points_mat = hstack([matrix(list(world_points[k]) + [1.0]).T for k in all_keys])
        image_points_mat = P * world_points_mat
        image_points_mat = matrix([[r[0,0]/r[0,2], r[0,1]/r[0,2]] for r in image_points_mat.T]).T
        util.draw_points(annotate_image,
                         dict(zip(all_keys, list(image_points_mat.T))))


    return K, R

if __name__ == "__main__":
    world_circles = util.get_circle_pattern(roll_radius=71.)

    optlist, args = getopt.getopt(sys.argv[1:], 'i:o:')

    in_file_name = None
    out_file_name = None
    for opt, param in optlist:
        if opt == "-i":
            in_file_name = param
        if opt == "-o":
            out_file_name = param

    if not in_file_name:
        raise Exception("Usage: %s -i <input image> [-o <output image>]" % sys.argv[0])

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
    K, R = solve(world_circles, image_circles, annotate_image=color_image)
    print K
    print R

    if out_file_name:
        cv.SaveImage(out_file_name, color_image)

