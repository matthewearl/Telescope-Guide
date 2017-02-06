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

def _get_contour_moments(contours, idx, gray_im):
    contour = contours[idx]
    
    mask = np.zeros(gray_im.shape)
    cv2.drawContours(mask, contours, idx, 1)
    
    m = cv2.moments(mask * gray_im)

    return m


def _get_stars_from_thr_im(thr_im, gray_im):
    _, contours, _ = cv2.findContours(thr_im.astype(np.uint8),
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
    for idx in range(len(contours)):
        m = _get_contour_moments(contours, idx, gray_im)
        flux = m["m00"]
        if m["m00"] != 0.:
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            id = "{},{}".format(int(x), int(y))

            yield stardb.ImageStar(id, (x, y), flux)


def get_stars_from_image(input_im):
    gray_im = (input_im.copy() if len(input_im.shape) == 0
                               else np.mean(input_im, axis=2)).astype(np.uint8)

    median_subbed_im = gray_im - cv2.medianBlur(gray_im, 55).astype(np.float)

    thr_im = 1.0 * (median_subbed_im > np.percentile(median_subbed_im, 99.))

    for star in _get_stars_from_thr_im(thr_im, gray_im):
        yield star


if __name__ == "__main__":
    from pprint import pprint 
    import sys

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(logging.StreamHandler())

    im = cv2.imread(sys.argv[1])
    pprint(get_stars_from_image(im))

