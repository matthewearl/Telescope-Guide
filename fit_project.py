#!/usr/bin/python

import getopt, sys, cv, math
import numpy.linalg
import scipy.linalg
from numpy import *
import find_circles
import util

def col_slice(M, cols):
    return hstack([M[:, i:(i+1)] for i in cols])

def left_inverse(A):
    return (A.T * A).I * A.T

def right_inverse(A):
    return A.T * (A * A.T).I

def solve(world_points_in, image_points_in, pixel_scale, annotate_image=None):
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

    Return: 4x4 matrix representing the camera's orientation, and a pixel
            pixel scale.
    """

    assert set(world_points_in.keys()) >= set(image_points_in.keys())
    keys = sorted(list(image_points_in.keys()))
    assert len(keys) >= 4

    world_points = hstack([matrix(list(world_points_in[k])).T for k in keys])
            
    # Choose a "good" set of 4 basis indices
    basis_indices = [0]
    basis_indices += [argmax([numpy.linalg.norm(world_points[:, i]) for i,k in enumerate(keys)])]
    def dist_from_line(idx):
        v = world_points[:, idx] - world_points[:, basis_indices[0]]
        d = world_points[:, basis_indices[1]] - world_points[:, basis_indices[0]]
        d = d / numpy.linalg.norm(d)
        v -= d * (d.T * v)[0, 0]
        return numpy.linalg.norm(v)
    basis_indices += [argmax([dist_from_line(i) for i,k in enumerate(keys)])]
    def dist_from_plane(idx):
        v = world_points[:, idx] - world_points[:, basis_indices[0]]
        a = world_points[:, basis_indices[1]] - world_points[:, basis_indices[0]]
        b = world_points[:, basis_indices[2]] - world_points[:, basis_indices[0]]
        d = matrix(cross(a.T, b.T).T)
        d = d / numpy.linalg.norm(d)
        return abs((d.T * v)[0, 0])
    basis_indices += [argmax([dist_from_plane(i) for i,k in enumerate(keys)])]

    basis        = hstack(world_points[:, i] for i in basis_indices)
    image_points = hstack([matrix(list(image_points_in[k]) + [pixel_scale]).T for k in keys])
    image_points = image_points / pixel_scale

    print [keys[i] for i in basis_indices]

    # Choose coeffs such that basis * coeffs = P
    # where P is world_points relative to the first basis vector
    def sub_origin(M):
        return M - hstack([basis[:, :1]] * M.shape[1])
    coeffs = sub_origin(basis[:, 1:]).I * sub_origin(world_points)

    # Compute matrix M and such that M * [z0, z1, z2, ... zN] = 0
    # where zi are proportional to the  Z-value of the i'th image point
    def M_for_image_point(idx):
        assert idx not in basis_indices

        out = matrix(zeros((3, len(keys))))

        # Set d,e,f st:
        #   d * (b[1] - b[0]) + e * (b[2] - b[0]) + f * (b[3] - b[0]) =
        #       world_points[idx]]
        d, e, f = [coeffs[i, key_idx] for i in [0,1,2]]

        out[:, basis_indices[0]:][:,:1] =  (1 - d - e - f) * image_points[:,basis_indices[0]:][:,:1]
        out[:, basis_indices[1]:][:,:1] =  d * image_points[:,basis_indices[1]][:,:1]
        out[:, basis_indices[2]:][:,:1] =  e * image_points[:,basis_indices[2]][:,:1]
        out[:, basis_indices[3]:][:,:1] =  f * image_points[:,basis_indices[3]][:,:1]
        out[:, idx:][:,:1]              = -image_points[:,idx][:,:1]

        return out
    M = vstack([M_for_image_point(key_idx)
                    for key_idx in xrange(len(keys))
                    if key_idx not in basis_indices])

    # Solve for Z by taking the eigenvector corresponding with the smallest
    # eigenvalue.
    eig_vals, eig_vecs = numpy.linalg.eig(M.T * M)
    Z = (eig_vecs.T[eig_vals.argmin()]).T

    # Project points. The scale of the projected points will be wrong, and the
    # orientation is still unknown.
    camera_points = matrix(array(image_points) * array(vstack([Z.T] * 3)))

    # Compute the rotation (and scale) from world space to camera space.
    P = (camera_points - hstack([camera_points[:,0:1]] * camera_points.shape[1])) * \
        right_inverse(world_points - hstack([world_points[:,0:1]] * world_points.shape[1]))

    K, R = map(matrix, scipy.linalg.rq(P))
    for i in xrange(3):
        if K[i,i] < 0:
            R[i:(i+1), :] = -R[i:(i+1), :]
            K[:, i:(i+1)] = -K[:, i:(i+1)]

    scale = 3.0 / sum(K[i,i] for i in xrange(3))
    t = scale * camera_points[:, basis_indices[0]:][:, :1] - R * world_points[:, basis_indices[0]:][:, :1]
    P = hstack((R, t))

    # Annotate the image, if we've been asked to do so.
    if annotate_image:
        all_keys = list(world_points_in.keys())
        world_points_mat = hstack([matrix(list(world_points_in[k]) + [1.0]).T for k in all_keys])
        image_points_mat = P * world_points_mat
        image_points_mat = matrix([[r[0,0]/r[0,2], r[0,1]/r[0,2]] for r in image_points_mat.T]).T
        image_points_mat *= pixel_scale
        util.draw_points(annotate_image,
                         dict(zip(all_keys, list(image_points_mat.T))))

    return P


def test_solve():
    pixel_scale = 1000.0
    world_points = matrix([[0.0, 0.0, 0.0, 1.0],
                           [20.0, 0.0, 0.0, 1.0],
                           [0.0, 15.0, 0.0, 1.0],
                           [0.0, 0.0, 10.0, 1.0],
                           [1.0, 1.0, 1.0, 1.0],
                           [2.0, 1.0, 1.0, 1.0],
                           [1.0, 2.0, 1.0, 1.0],
                           [1.0, 1.0, 2.0, 1.0]]).T
    theta = 1.5
    s = math.sin(theta)
    c = math.cos(theta)
    R = matrix([[ c,   -s, 0.0],
                [ s,    c, 0.0],
                [0.0, 0.0, 1.0]])
    P = hstack([R, matrix([[0., 0., 30.]]).T])
    print P
    image_points = P * world_points
    from pprint import pprint as pp
    pp(world_points)
    pp(image_points)
    image_points = matrix([[r[0,0]/r[0,2], r[0,1]/r[0,2]] for r in image_points.T]).T
    image_points = 1000.0 * image_points

    keys = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    

    world_points = dict(zip(keys, [tuple(p) for p in array(world_points[:3, :].T)]))
    image_points = dict(zip(keys, [tuple(p) for p in array(image_points.T)]))

    print solve(world_points, image_points,
                pixel_scale=pixel_scale)

if __name__ == "__main__":
    #numpy.set_printoptions(precision=3)
    #test_solve()
    #sys.exit(0)

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
    R = solve(world_circles,
                 image_circles,
                 pixel_scale=3182.4,
                 annotate_image=color_image)
    print R

    if out_file_name:
        cv.SaveImage(out_file_name, color_image)

