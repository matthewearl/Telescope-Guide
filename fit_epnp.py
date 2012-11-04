#!/usr/bin/python

import getopt, sys
import cv
import numpy.linalg
from numpy.linalg import norm
from numpy import *
import find_circles, util

def sub_all(M, c):
    return vstack(col - c.T for col in M.T).T

def make_coeff_matrix(world_points, ctrl_indices):
    """
    Return a matrix C such that:
        world_points.T = C * control_points.T

    With each row of C summing to 1.
    """

    # First find coeffs r, s, t to satisfy:
    #    w - c[0] = r(c[1] - c[0]) + s(c[2] - c[0]) + t(c[3] - c[0])
    # Then rearrange to get: 
    #    w = (1 - r - s - t)*c[0] + r*c[1] + s*c[2] + t*c[3]
    control_points = util.col_slice(world_points, ctrl_indices)

    world_points = sub_all(world_points, control_points[:, 0:1])
    control_points = sub_all(control_points[:, 1:], control_points[:, 0:1])
    C = world_points.T * util.right_inverse(control_points.T)

    return matrix([[1. - r - s -t, r, s, t] for r, s, t in array(C)])

def test_make_coeff_matrix(world_points, ctrl_indices):
    C = make_coeff_matrix(world_points, ctrl_indices)

    control_points = util.col_slice(world_points, ctrl_indices)

    print "C = %s" % C
    print "world_points.T = %s" % world_points.T 
    print "C*control_points.T = %s" % (C * control_points.T)

def choose_control_points(world_points):
    indices = [0]
    indices += [argmax([numpy.linalg.norm(world_points[:, i]) for i in xrange(world_points.shape[1])])]
    def dist_from_line(idx):
        v = world_points[:, idx] - world_points[:, indices[0]]
        d = world_points[:, indices[1]] - world_points[:, indices[0]]
        d = d / numpy.linalg.norm(d)
        v -= d * (d.T * v)[0, 0]
        return numpy.linalg.norm(v)
    indices += [argmax([dist_from_line(i) for i in xrange(world_points.shape[1])])]
    def dist_from_plane(idx):
        v = world_points[:, idx] - world_points[:, indices[0]]
        a = world_points[:, indices[1]] - world_points[:, indices[0]]
        b = world_points[:, indices[2]] - world_points[:, indices[0]]
        d = matrix(cross(a.T, b.T).T)
        d = d / numpy.linalg.norm(d)
        return abs((d.T * v)[0, 0])
    indices += [argmax([dist_from_plane(i) for i in xrange(world_points.shape[1])])]

    return indices

def make_M(image_points, C):
    def make_single(i, j):
        return matrix([[C[i,j], 0.0, -C[i, j] * image_points[0, i]],
                       [0.0, C[i,j], -C[i, j] * image_points[1, i]]])
    def make_row(i):
        return hstack([make_single(i, j) for j in xrange(4)])

    return vstack([make_row(i) for i in xrange(image_points.shape[1])])

def calc_reprojection_error(R, offs, world_points, image_points):
    reprojected = R * world_points + hstack([offs] * world_points.shape[1])
    err = (reprojected - image_points).flatten()
    return (err * err.T)[0, 0]

def calc_beta_case_1(V, ctrl_points):
    num = sum(norm(V[(3*i):(3*i+3),:] - V[(3*j):(3*j+3),:]) * norm(ctrl_points[:,i] - ctrl_points[:,j]) for i in xrange(4) for j in xrange(4))
    den = sum(norm(V[(3*i):(3*i+3),:] - V[(3*j):(3*j+3),:])**2 for i in xrange(4) for j in xrange(4))
    return matrix([[num/den]])

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
        T = hstack([V[(3*i):(3*i+3),0:1] - V[(3*j):(3*j+3), 0:1],
                    V[(3*i):(3*i+3),1:2] - V[(3*j):(3*j+3), 1:2]])
        return coeff_sqr_sum(T)

    pair_indices = [(i, j) for i in xrange(4) for j in xrange(i+1, 4)]

    L = vstack([calc_row(i, j) for i, j in pair_indices])
    rho = matrix([[norm(ctrl_points[:,i] - ctrl_points[:,j])**2 for i, j in pair_indices]]).T
    beta = util.left_inverse(L) * rho
    
    return matrix([math.sqrt(beta[0,0]), math.sqrt(beta[2,0])])

def calc_beta_case_3(V, ctrl_points):
    def coeff_sqr(a):
        """
        Return c st:
            c * [b1^2, b1*b2, b1*b3, b2^2, b2*b3, b3^2].T 
                   i = ( a * [b1, b2, b3].T )^2
        """
        return matrix([[a[0,0]**2,            # b1^2
                       2. * a[0,0] * a[0,1],  # b1 * b2
                       2. * a[0,0] * a[0,2],  # b1 * b3
                       a[0,1]**2,             # b2^2
                       2. * a[0,1] * a[0,2],  # b2 * b3
                       a[0,2]**2]])           # b^3

    def coeff_sqr_sum(M):
        return sum(coeff_sqr(r) for r in M)

    def calc_row(i, j):
        """
        Calculate a row r such that r * [b1^2, b1*b2, b2^2].T ==
        || ctrl_points[i] - ctrl_points[j] ||^2
        """

        # Calc T such that || T * [b1, b2, b3].T ||^2 = 
        #   || ctrl_points[i] - ctrl_points[j] ||^2
        T = hstack([V[(3*i):(3*i+3),0:1] - V[(3*j):(3*j+3), 0:1],
                    V[(3*i):(3*i+3),1:2] - V[(3*j):(3*j+3), 1:2],
                    V[(3*i):(3*i+3),2:3] - V[(3*j):(3*j+3), 2:3]])
        return coeff_sqr_sum(T)

    pair_indices = [(i, j) for i in xrange(4) for j in xrange(i+1, 4)]

    L = vstack([calc_row(i, j) for i, j in pair_indices])
    rho = matrix([[norm(ctrl_points[:,i] - ctrl_points[:,j])**2 for i, j in pair_indices]]).T
    beta = L.I * rho
    
    return matrix([math.sqrt(beta[0,0]), math.sqrt(beta[3,0]), math.sqrt(beta[5,0])])

def solve(world_points_in, image_points_in, pixel_scale, annotate_image=None):
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
    V = eig_vecs.T[eig_vals.argsort()].T

    world_ctrl_points = util.col_slice(world_points, control_indices)
    b1 = calc_beta_case_1(V[:, :1], world_ctrl_points)
    b2 = calc_beta_case_2(V[:, :2], world_ctrl_points)
    b3 = calc_beta_case_3(V[:, :3], world_ctrl_points)
   
    outs = []
    errs = []
    for b in [b1, b2, b3]:
        x = V[:, :b.shape[1]] * b.T
        x = x.reshape((4, 3))

        R, offs = util.orientation_from_correspondences(world_ctrl_points, x.T)
        outs.append((R, offs))

        e = calc_reprojection_error(R, offs, world_points, image_points)
        print "Reprojection error = %f" % e
        errs.append(e)
    
    return outs[array(errs).argmin()]

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
    image_circles = {'12a': (-317.5694384188557, 49.75731612763366), '12b': (-96.00539504774679, 84.90609863144823), '9a': (-244.34532303231458, -423.2478097015005), '9b': (-16.627520506430983, -428.16211440291045), '16a': (271.666915194201, -541.9738592210165), '16b': (573.9470467983265, -557.8967418871907), '20a': (83.16644564441776, 112.82609270004014), '20b': (381.67270013250027, 159.84200953833442), '1a': (-557.6921250392775, -415.6084477626862), '10b': (-90.35616570745606, -274.40682452027295), '1b': (-383.11150411227527, -420.064586005451), '3b': (-475.49645931669147, -126.59631671223474), '3a': (-644.811504223775, -142.71317855707593), '13b': (-29.32539202636599, 229.24184165572365), '13a': (-250.6791984453264, 183.45648074060068), '10a': (-315.0212557770142, -281.1252405701571), '11a': (-340.5356728299041, -113.81499057163956), '11b': (-118.06984822255208, -92.51222867116485), '8b': (87.04633444530486, -531.5555016927228), '8a': (-143.23542414856684, -519.3916219057564), '17b': (471.0422707491748, -437.53405915568555), '17a': (166.47626524629277, -432.04570198629153), '21b': (443.59446639584985, 325.62811019594596), '21a': (148.58915059685432, 265.467437792056), '19a': (62.402457031310405, -75.58991161062545), '19b': (363.915065217152, -46.78180312087238), '18b': (395.2352132422966, -259.21992781257404), '18a': (91.63190150970695, -268.9890004930937), '2a': (-622.6476835944959, -289.852633203142), '2b': (-451.438997287662, -285.0906225633994), '4b': (-452.28270616470945, 28.406999641906623), '4a': (-621.1682101666386, 1.0951753166343678)}
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

