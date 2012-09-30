#!/usr/bin/python

import numpy.linalg
from numpy.linalg import norm
from numpy import *
import find_circles

def make_coeff_matrix(world_points, ctrl_indices):
    """
    Return a matrix C such that:
        world_points.T = C * control_points.T

    With each row of C summing to 1.
    """

    # First find coeffs a, b, c to satisfy:
    #    w - c[0] = r(c[1] - c[0]) + s(c[2] - c[0]) + t(c[3] - c[0])
    # Then rearrange to get: 
    #    w = (1 - r - s - t)*c[0] + r*c[1] + s*c[2] + t*c[3]
    control_points = util.col_slice(world_points, ctrl_indices)

    def sub_all(M, c):
        return hstack(col - c.T for col in control_points.T).T
    control_points = sub_all(control_points[:, 1:], control_points[:, 0:1])
    world_points = sub_all(world_points, control_points[:, 0:1])
    C = world_points.T * util.right_inverse(control_points):

    return matrix([[1. - r - s -t, r, s, t] for r, s, t in array(C)])

def choose_control_points(world_points):
    indices = [0]
    indices += [argmax([numpy.linalg.norm(world_points[:, i]) for i,k in enumerate(keys)])]
    def dist_from_line(idx):
        v = world_points[:, idx] - world_points[:, indices[0]]
        d = world_points[:, indices[1]] - world_points[:, indices[0]]
        d = d / numpy.linalg.norm(d)
        v -= d * (d.T * v)[0, 0]
        return numpy.linalg.norm(v)
    indices += [argmax([dist_from_line(i) for i,k in enumerate(keys)])]
    def dist_from_plane(idx):
        v = world_points[:, idx] - world_points[:, indices[0]]
        a = world_points[:, indices[1]] - world_points[:, indices[0]]
        b = world_points[:, indices[2]] - world_points[:, indices[0]]
        d = matrix(cross(a.T, b.T).T)
        d = d / numpy.linalg.norm(d)
        return abs((d.T * v)[0, 0])
    indices += [argmax([dist_from_plane(i) for i,k in enumerate(keys)])]

def make_M(image_points, C):
    def make_single(i, j):
        return matrix([[C[i,j], 0.0, -C[i, j] * image_points[0, i]],
                       [0.0, C[i,j], -C[i, j] * image_points[1, i]]])
    def make_row(i):
        hstack([make_single(i, j) for j in xrange(4)])

    return vstack([make_row(i) for i in xrange(image_points.shape[1])])

def calc_beta_case_1(V, ctrl_points):
    num = sum(norm(V[(3*i):(3*i+3),:] - V[(3*j):(3*j+3),:]) * norm(ctrl_points[:,i] - ctrl_points[:,j]) for i in xrange(4) for j in xrange(4))
    den = sum(norm(V[(3*i):(3*i+3),:] - V[(3*j):(3*j+3),:])**2 for i in xrange(4) for j in xrange(4))
    return num/den

def calc_beta_case_2(V, ctrl_points):
    def coeff_sqr(a):
        """
        Return c st:
            c * [b1^2, b1*b2, b2^2].T = ( a * [b1, b2].T )^2
        """
        return matrix([[a[0,0]**2, 2*a[0,0]*a[0,1], a[0,1]**2]])

    def coeff_sqr_sum(M):
        return sum(coeff_sqr(r) for r in M)

    def calc_row(i, j):
        """
        Calculate a row r such that r * [b1^2, b1*b2, b2^2].T ==
        || ctrl_points[i] - ctrl_points[j] ||^2
        """

        # Calc T such that || T * [b1, b2].T ||^2 = 
        #   || ctrl_points[i] - ctrl_points[j] ||^2
        T = hstack([v[(3*i):(3*i+3),0:1] - v[(3*j):(3*j+3), 0:1],
                    v[(3*i):(3*i+3),1:2] - v[(3*j):(3*j+3), 1:2]])
        return coeff_sqr_sum(M)

    return vstack([calc_row(i, j) for i in xrange(4) for j in xrange(i+1, 4)])

def solve(world_points, image_points, pixel_scale, annotate_image=None):
    """
    Find a camera's orientation given a set of world coordinates and
    corresponding set of camera coordinates.

    world_points: Dict mapping point names to triples corresponding with world
                  x, y, z coordinates.
    image_points: Dict mapping point names to triples corresponding with
                  camera x, y coordinates. Coordinates are translated such that
                  0, 0 corresponds with the centre of the image.

    Return: 4x4 matrix representing the camera's orientation.
    """

    assert set(world_points_in.keys()) >= set(image_points_in.keys())
    keys = sorted(list(image_points_in.keys()))
    assert len(keys) >= 4
    world_points = hstack([matrix(list(world_points_in[k])).T for k in keys])
    image_points = hstack([matrix(list(image_points_in[k]) + [pixel_scale]).T for k in keys])
    image_points = image_points / pixel_scale

    control_indices = choose_control_points(world_points)
    C = make_coeff_matrix(world_points, control_indices)
    M = make_M(image_points, C)

    eig_vals, eig_vecs = numpy.linalg.eig(M.T * M)
    V = eig_vecs.T[eigvals.argsort()]

    b1 = calc_beta_case_1(V[0:1, :])
    b2 = calc_beta_case_2(V[0:2, :])
    b3 = calc_beta_case_3(V[0:3, :])
   

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

