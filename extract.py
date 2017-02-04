#!/usr/bin/env python

__all__ = (
    "get_stars_from_image",
)

import json
import logging
import random

import cv2
import numpy as np

import stardb

LOG = logging.getLogger(__name__)


def _get_stars_from_thr_im(thr_im):
    _, contours, _ = cv2.findContours(thr_im.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        m = cv2.moments(contour)
        flux = m["m00"]
        if flux != 0:
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            id = "{},{}".format(int(x), int(y))

            yield stardb.ImageStar(id, (x, y), flux)


def get_stars_from_image(input_im):
    im_gray = (input_im.copy() if len(input_im.shape) == 0
                               else np.mean(input_im, axis=2)).astype(np.uint8)

    median_subbed_im = im_gray - cv2.medianBlur(im_gray, 35).astype(np.float)

    thr_im = 1.0 * (median_subbed_im > np.percentile(median_subbed_im, 99.))

    for star in _get_stars_from_thr_im(thr_im):
        yield star


if __name__ == "__main__":
    from pprint import pprint 
    import sys

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(logging.StreamHandler())

    im = cv2.imread(sys.argv[1])
    pprint(get_stars_from_image(im))

