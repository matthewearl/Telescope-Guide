#!/usr/bin/python

from numpy import *
import util

__all__ = ['solve']

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
        world_points_mat = hstack([matrix(list(world_points[k]) + [1.0]).T for k in keys])
        image_points_mat = P * world_points_mat
        util.draw_points(annotate_image,
                         dict(zip(keys, list(image_points_mat.T))))

    return P

if __name__ == "__main__":
    world_circles = util.get_circle_pattern()

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
    print solve(world_circles, image_circles, annotate_image=color_image)

    if out_file_name:
        cv.SaveImage(out_file_name, color_image)

