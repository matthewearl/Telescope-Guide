import getopt
import find_circles
import gen_target
import cv
import util
from numpy import *

__all__ = ['fitter_main', 'Fitter']

class Fitter(object):
    def solve(self,
              world_points,
              image_features,
              pixel_scale=None,
              annotate_image=None):
        raise NotImplementedError()

def camera_to_image(m, ps):
    return ps * matrix([[c[0, 0] / c[0, 2], c[0, 1] / c[0, 2]] for c in m.T]).T

def project_points(R, T, pixel_scale, world_points, keys):
    world_points = hstack([matrix(list(world_points[k])).T for k in keys])
    camera_points = R * world_points + hstack([T] * world_points.shape[1])

    return camera_to_image(camera_points, pixel_scale)

def calc_reprojection_error(R,
                            T,
                            pixel_scale,
                            world_points,
                            image_features):
    assert set(world_points.keys()) >= set(image_features.keys())
    keys = sorted(list(image_features.keys()))

    reprojected = project_points(R, T, pixel_scale, world_points, keys)

    image_points = hstack([matrix(list(image_features[k].get_centre())).T for k in keys])

    err = (reprojected - image_points).flatten()

    return (err * err.T)[0, 0]

def draw_reprojected(R, T, pixel_scale, world_points, annotate_image):
    keys = list(world_points.keys())
    reprojected = project_points(R, T, pixel_scale, world_points, keys)

    for key in keys:
        util.draw_points(annotate_image,
                         dict(zip(keys, list(reprojected.T))))


def fitter_main(args, fitter):
    world_points = gen_target.get_targets()

    optlist, _ = getopt.getopt(args[1:], 'i:o:')

    in_file_name = None
    out_file_name = None
    for opt, param in optlist:
        if opt == "-i":
            in_file_name = param
        if opt == "-o":
            out_file_name = param

    if not in_file_name:
        raise Exception("Usage: %s -i <input image> [-o <output image>]" % args[0])

    print "Loading image (black and white)"
    image = cv.LoadImage(in_file_name, False)
    print "Loading image (colour)"
    color_image = cv.LoadImage(in_file_name, True)
    print "Finding labelled circles"
    image_features = find_circles.find_labelled_circles(image,
                                                        annotate_image=color_image,
                                                        find_ellipses=True)
    print "Solving"
    R, T, pixel_scale = fitter.solve(world_points,
                                     image_features,
                                     pixel_scale=2986.3,
                                     annotate_image=color_image)
    print R, T, pixel_scale

    err = calc_reprojection_error(R, T, pixel_scale, world_points, image_features)
    print "Reprojection error: %f" % err

    draw_reprojected(R, T, pixel_scale, world_points, color_image)

    if out_file_name:
        cv.SaveImage(out_file_name, color_image)
