#!/usr/bin/env python

import argparse
import collections
import itertools
import logging
import math
import subprocess
import time

from numpy import *
import scipy.spatial

import camera
import stardb
import util


LOG = logging.getLogger(__name__)

# Each asterism consists of one star and NUM_NEIGHBOURS other stars.
NUM_NEIGHBOURS = 3


class CouldNotAlignError(Exception):
    pass

class Asterism(object):
    def __init__(self, main_star, neighbours):
        assert len(neighbours) >= 3

        self.main_star = main_star
        self.neighbours = neighbours
        
        # Calculate the neighbour vectors and distances relative to the main
        # star. These lists are used throughout this function.
        diffs = [neigh.vec - main_star.vec for neigh in neighbours]
        dists = [linalg.norm(d) for d in diffs]

        # Determine the index of the farthest neighbour, and the indices of
        # all the other neighbours.
        max_idx = max(enumerate(dists), key=(lambda t: t[1]))[0]
        other_idx = range(0,max_idx) + range(max_idx+1,len(neighbours))

        # Establish a reference frame based on the relative position of the
        # main star from the farthest star.
        coord_frame = matrix(zeros((3,3)))
        coord_frame[:, 2] = main_star.normal
        coord_frame[:, 0] = cross(diffs[max_idx].T, coord_frame[:, 2].T).T
        coord_frame[:, 1] = cross(coord_frame[:,0].T, coord_frame[:,2].T).T
        for i in range(3):
            coord_frame[:, i] /= linalg.norm(coord_frame[:, i])

        # Compute the positions of the other stars in this reference frame.
        # This will normalise for rotation.  Divide by the distance of the
        # furthest star to normalise for scale.  Ignore the Z-axis value (which
        # will be approximately 1).
        hash_matrix = ((1. / dists[max_idx]) * coord_frame.I *
                        hstack(diffs[idx] for idx in other_idx))
        hash_matrix = hash_matrix[:2, :]

        # Normalise the column ordering. (The choice of ordering by X-axis
        # value is fairly arbitrary.)
        hash_matrix = hash_matrix[:, array(hash_matrix)[0, :].argsort()]

        # Compose the final 1-D descriptor for this asterism.
        self.vec = hash_matrix.T.reshape((-1, 1))

    def __repr__(self):
        return "<Asterism(main_star=%r, neighbours=%r, vec=%r)>" % (
                        self.main_star, self.neighbours, self.vec)

class _NoAsterism:
    pass

def asterism_for_star(main_star,
                      star_db,
                      num_neighbours=NUM_NEIGHBOURS,
                      start_radius=math.pi / 180,
                      max_radius=20 * math.pi / 180):
    radius = start_radius
    neighbours = []
    while len(neighbours) < num_neighbours and radius <= max_radius:
        neighbours = [(s, d) for s, d in
                                star_db.search_vec(main_star.vec, radius)
                           if s.mag < main_star.mag]
        radius *= 2.

    if len(neighbours) < num_neighbours:
        raise _NoAsterism

    neighbours = sorted(neighbours, key=lambda (s, d): d)[:num_neighbours]

    return Asterism(main_star, [s for s, d in neighbours])


def asterisms_gen(star_db, mag_limit=8.0):
    i = 0
    num_asterisms = 0
    for star in (s for s in star_db if s.mag < mag_limit):
        if i % 5000 == 0:
            LOG.info("%s / %s. %s asterisms.",
                     i, len(star_db), num_asterisms)
        i += 1
        try:
            yield asterism_for_star(star, star_db)
            num_asterisms += 1
        except _NoAsterism:
            pass
    LOG.info("%s asterisms", num_asterisms)


class AsterismDatabase(object):
    def __init__(self, asterism_iterable):
        self.asterisms = list(asterism_iterable)
        self.asterism_dict = collections.defaultdict(set)
        for asterism in self.asterisms:
            self.asterism_dict[asterism.main_star.id].add(asterism)
        self.tree = scipy.spatial.cKDTree(vstack([ast.vec.T
                                                   for ast in self.asterisms]))

    def search(self, query_ast):
        dist, idx = self.tree.query(query_ast.vec.flat)
        return self.asterisms[idx], dist


def _calibrate_image_stars2(image_stars, ast_db, brightest_n=10):
    bright_im_stars = sorted(image_stars, key=lambda s: s.mag)[:brightest_n]

    for stars in itertools.combinations(bright_im_stars, 4):
        for i in range(3):
            image_ast = Asterism(stars[i], stars[:i] + stars[i + 1:])
            ast, dist = ast_db.search(image_ast)

            yield image_ast, ast, dist


def _calibrate_image_stars(image_stars, ast_db):
    image_star_db = stardb.StarDatabase(image_stars)

    # Obtain a (reasonable) upper bound on maximum distance between any two
    # image stars.
    radius = linalg.norm(amax(hstack(s.vec for s in image_stars), axis=1))

    for image_star in image_star_db:
        try:
            image_ast = asterism_for_star(image_star,
                                          image_star_db,
                                          start_radius=radius,
                                          max_radius=radius)
        except _NoAsterism:
            LOG.debug("No asterism could be generated for %r", image_star)
            pass
        else:
            ast, dist = ast_db.search(image_ast)
            LOG.debug("Matched %r with %r, distance = %r",
                      image_star, ast.main_star, dist)
            yield image_ast, ast, dist


def draw_stars(star_db, image, R, cam_model, mag_limit=4.0):
    """
    Draw stars on an image. The camera is assumed to be orientated at R.
    """
    import cv

    for star in (s for s in star_db if s.mag < mag_limit):
        vec = R.T * star.vec
        if vec[2, 0] > 0:
            coords = tuple(map(int, cam_model.vec_to_pixel(vec)))
            cv.Circle(image, coords, 5, cv.CV_RGB(0, 255, 0))


class StarAlignArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(StarAlignArgumentParser, self).__init__(description='Align a star field')

        self.add_argument('-i', '--input-image', help='Input image', required=True)
        self.add_argument('-o', '--output-image', help='Output image', required=True)
        self.add_argument('-a', '--xy-list', help='XY list')
        self.add_argument('-c', '--cat-file', help='SExtractor catalogue')
        self.add_argument('-p',
                          '--pixel-scale',
                          type=float,
                          help='Pixel scale',
                          required=True)
        self.add_argument('-b',
                          '--barrel-distortion',
                          type=float,
                          help='Barrel distortion',
                          required=True)

if __name__ == "__main__":
    import cv
    args = StarAlignArgumentParser().parse_args()

    if args.xy_list is None and args.cat_file is None:
        print "%f: Generating augmented xy-list" % time.clock()
        args.xy_list = "temp.axy"
        rc = subprocess.call(["augment-xylist",
                              "-i",
                              args.input_image,
                              "-o",
                              args.xy_list])
        if rc != 0:
            raise Exception("augment-xylist returned %u" % rc)

    print "%f: Loading input image" % time.clock()
    image = cv.LoadImage(args.input_image, True)

    print "%f: Building star database..." % time.clock()
    star_db = stardb.StarDatabase(stardb.hip_star_gen('data/hip_main.dat'))

    print "%f: Building asterism database..." % time.clock()
    ast_db = AsterismDatabase(asterisms_gen(star_db))

    print "%f: Aligning image" % time.clock()
    ps = args.pixel_scale * min(image.width, image.height) / 2592.
    bd = args.barrel_distortion * (2592. / min(image.width, image.height))**2
    cam_model = camera.BarrelDistortionCameraModel(ps, bd, image.width, image.height)
    if args.xy_list is not None:
        image_stars = stardb.xy_list_star_gen(args.xy_list, cam_model)
    elif args.cat_file is not None:
        image_stars = stardb.cat_star_gen(args.cat_file, cam_model)
    else:
        raise Exception("No xy points specified")
    
    ra, dec, R = align_image2(image_stars, ast_db)
    print "RA: %s, Dec: %s" % (stardb.ra_to_str(ra), stardb.dec_to_str(dec))

    print "%f: Drawing stars" % time.clock()
    draw_stars(star_db, image, R, cam_model)

    print "%f: Saving image" % time.clock()
    cv.SaveImage(args.output_image, image)

    print "%f: Done" % time.clock()

