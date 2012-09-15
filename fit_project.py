#!/usr/bin/python

import getopt, sys, cv
import numpy.linalg
import scipy.linalg
from numpy import *
import find_circles
import util

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
    keys = list(image_points_in.keys())
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
    print "Basis points: %s" % ([keys[idx] for idx in basis_indices])

    basis        = hstack(world_points[:, i] for i in basis_indices)
    image_points = hstack([matrix(list(image_points_in[k]) + [pixel_scale]).T for k in keys])
    image_points = image_points / pixel_scale

    print "Basis: %s" % basis

    # Choose coeffs such that basis * coeffs = P
    # where P is world_points relative to the first basis vector
    def sub_origin(M):
        return M - hstack([basis[:, 0]] * M.shape[1])
    coeffs = left_inverse(sub_origin(basis[:, 1:])) * sub_origin(world_points)

    # Compute matrix M and such that M * [z0, z1, z2, ... zN] = 0
    # where zi are proportional to the  Z-value of the i'th image point
    def M_for_image_point(idx):
        out = matrix(zeros((3, len(keys))))

        # Set d,e,f st:
        #   d * (b[1] - b[0]) + e * (b[2] - b[0]) + f * (b[3] - b[0]) =
        #       world_points[idx]]
        d, e, f = [coeffs[i, key_idx] for i in [0,1,2]]

        out[:, basis_indices[0]:][:,:1] =  (1 - d - e - f) * image_points[:,basis_indices[0]:][:,:1]
        out[:, basis_indices[1]:][:,:1] =  d * image_points[:,basis_indices[1]][:,:1]
        out[:, basis_indices[2]:][:,:1] =  e * image_points[:,basis_indices[2]][:,:1]
        out[:, basis_indices[3]:][:,:1] =  f * image_points[:,basis_indices[3]][:,:1]
        out[:, idx:][:,:1]              += -image_points[:,idx][:,:1]

        return out
    M = vstack([M_for_image_point(key_idx) for key_idx in xrange(len(keys))])

    # Solve for Z by taking the eigenvector corresponding with the smallest
    # eigenvalue.
    eig_vals, eig_vecs = numpy.linalg.eig(M.T * M)
    Z = (eig_vecs.T[eig_vals.argmin()]).T

    # Project points. The scale of the projected points will be wrong, and the
    # orientation is still unknown.
    camera_points = matrix(array(image_points) * array(vstack([Z.T] * 3)))
    
    # Compute the 3x4 matrix mapping world points onto scaled camera space points.
    world_points_4 = vstack([world_points, matrix([1.] * world_points.shape[1])])
    P = camera_points * right_inverse(world_points_4)

    # Split into rotation, translation, and intrinsic parameter matrices.
    # The intrinsic parameters should just be the scale factor * I.
    K, R = map(matrix, scipy.linalg.rq(P[:, :3]))
    t = K.I * P[:, 3:]
    R = hstack((R, t))

    # Annotate the image, if we've been asked to do so.
    if annotate_image:
        all_keys = list(world_points.keys())
        world_points_mat = hstack([matrix(list(world_points[k]) + [1.0]).T for k in all_keys])
        image_points_mat = P * world_points_mat
        image_points_mat = matrix([[r[0,0]/r[0,2], r[0,1]/r[0,2]] for r in image_points_mat.T]).T
        image_points_mat *= pixel_scale
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
    #image_circles = find_circles.find_labelled_circles(image,
    #                                                   annotate_image=color_image,
    #                                                   centre_origin=True)
    image_circles = {'12a': (-98.0007336804506, -106.5496278426076), '12b': (-108.89069694634145, 188.12794690151895), '9a': (414.028757991573, -278.800932410119), '10a': (214.68538025935868, -268.2685703721377), '16a': (695.759577230478, 300.32145316527556), '16b': (767.5186245889404, 705.1516917963072), '20a': (-117.02852350851435, 426.98347570164844), '20b': (-130.58656611533752, 824.9837658988295), '1a': (360.7129962734425, -675.7609005698027), '10b': (233.93296139367612, 28.55036325866331), '1b': (390.29936549347894, -456.38883752216134), '9b': (452.69399880802393, 18.105342024327), '11a': (34.170686692925074, -205.4399628565518), '11b': (35.54704825653221, 90.96631969582631), '17b': (533.9528855995522, 672.6882144634903), '17a': (483.7013859354697, 261.34382447627513), '19a': (37.12702430059812, 333.17836453152677), '19b': (39.19530958007931, 740.2170329338646), '18b': (274.97420250911864, 684.4917766846534), '18a': (249.7207560486795, 272.5570474888916), '2a': (188.14760501190722, -664.6436969560143), '2b': (202.95593600586335, -445.5802326874091), '4b': (-91.2839593443482, -284.76357352822424), '4a': (-82.44614893280277, -507.0350293488773)}
    print image_circles
    print "Solving"
    K, R = solve(world_circles,
                 image_circles,
                 pixel_scale=3182.4,
                 annotate_image=color_image)
    print K
    print R

    if out_file_name:
        cv.SaveImage(out_file_name, color_image)

