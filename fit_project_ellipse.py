#!/usr/bin/python

import fit
import sys
import util
from numpy import *

class EllipseProjectFitter(fit.Fitter):
    def __init__(self):
        pass

    def solve(self,
              world_points,
              image_features,
              pixel_scale=None,
              annotate_image=None):

        assert pixel_scale is not None
        assert set(world_points.keys()) >= set(image_features.keys())
        keys = sorted(list(image_features.keys()))

        world_points = hstack([matrix(list(world_points[k])).T for k in keys])

        def project_feature(feature):
            centre = feature.get_centre()
            axes = feature.get_axes()
            assert axes[0] >= axes[1]

            x = centre[0] * 10.0  / axes[0]
            y = centre[1] * 10.0  / axes[0]
            z = 10.0 * pixel_scale / axes[0]

            out = matrix([[x], [y], [z]])
            return out

        projected_points = hstack([project_feature(image_features[k]) for k in keys])

        R, T = util.orientation_from_correspondences(world_points,
                                                     projected_points)

        return R, T, pixel_scale
                                                
if __name__ == "__main__":
    fit.fitter_main(sys.argv, EllipseProjectFitter())

