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


def _get_threshold_im(im, pct, dilation_amount):
    thr_im = 255 * (im > np.percentile(im, pct))

    thr_im = thr_im.astype(np.uint8)
    thr_im = cv2.dilate(thr_im, np.ones((dilation_amount, dilation_amount)))

    return thr_im


def _get_stars_from_thr_im(thr_im, min_blob_size, max_blob_size):
    _, contours, _ = cv2.findContours(thr_im,
                                      cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)

    for contour in contours:
        m = cv2.moments(contour)
        if min_blob_size <= m["m00"] <= max_blob_size:
            flux = m["m00"]
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            id = "{},{}".format(int(x), int(y))

            yield stardb.ImageStar(id, (x, y), flux)


def _get_stars_from_gray_im(im,
                            start_pct,
                            dilation_amount,
                            min_blob_size,
                            max_blob_size):
    pct = 100 - start_pct

    star_count = None
    while star_count is None or star_count > 50:
        LOG.debug("Trying threshold of %s", 100. - pct)
        cv2.imwrite("foo1.png", im)
        thr_im = _get_threshold_im(im, 100. - pct, dilation_amount)
        cv2.imwrite("foo2.png", thr_im)

        stars = list(_get_stars_from_thr_im(thr_im,
                                            min_blob_size,
                                            max_blob_size))
        star_count = len(stars)
        LOG.debug("Star count: %s", star_count)
        pct *= 0.9

    return stars


def get_stars_from_image(input_im,
                         dilation_amount,
                         min_blob_size,
                         max_blob_size):
    im_gray = (input_im.copy() if len(input_im.shape) == 0
                               else np.mean(input_im, axis=2))

    return _get_stars_from_gray_im(im_gray, 98.,
                                   dilation_amount,
                                   min_blob_size,
                                   max_blob_size)


if __name__ == "__main__":
    from pprint import pprint 
    import sys

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(logging.StreamHandler())

    im = cv2.imread(sys.argv[1])
    pprint(get_stars_from_image(im))

