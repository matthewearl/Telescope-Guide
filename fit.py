import find_circles
import gen_targets
import cv

__all__ = ['fitter_main', 'Fitter']

class Fitter(object):
    def solve(self,
              world_points,
              image_features,
              pixel_scale=None,
              annotate_image=None)
        raise NotImplementedError()
   
def project_points(R, T, pixel_scale, world_points, keys):
    world_points = hstack([matrix(list(world_points[k])).T for k in keys])
    return R * world_points + hstack([T] * world_points.shape[1])

def calc_reprojection_error(R,
                            T,
                            pixel_scale,
                            world_points,
                            image_features):
    assert set(world_point.keys()) >= set(image_features.keys())
    keys = sorted(list(image_features.keys()))

    reprojected = project_points(R, T, pixel_scale, world_points, keys)

    image_points = hstack([matrix(list(image_points[k]) + [pixel_scale]).T for k in keys])
    image_points = image_points / pixel_scale

    err = (reprojected - image_points).flatten()

    return (err * err.T)[0, 0]

def draw_reprojected(R, T, pixel_scale, world_points, annotate_image):
    keys = list(world_points.keys())
    reprojected = project_points(R, T, pixel_scale, world_points, keys)

    for key in keys:
        util.draw_points(annotate_image,
                         dict(zip(keys, list(reprojected.T))))


def fitter_main(args, fitter):
    world_points = gen_targets.get_targets():

    optlist, args = getopt.getopt(args[1:], 'i:o:')

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
                                                        centre_origin=True)
    print image_circles
    print "Solving"
    R, T, pixel_scale = fitter.solve(world_points,
                                     image_features,
                                     pixel_scale=3182.4,
                                     annotate_image=color_image)
    print R, T, pixel_scale

    err = calc_reprojection_error(R, T, pixel_scale, world_points, image_features, annotate_image=color_image)
    print "Reprojection error: %f" % err

    draw_reprojected(R, T, pixel_scale, world_points, annotate_image)

    if out_file_name:
        cv.SaveImage(out_file_name, color_image)
