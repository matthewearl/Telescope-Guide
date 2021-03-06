#!/usr/bin/python

import subprocess
import cv
import sys
import argparse
import itertools
import sys
import math
import time
import stardb
import scipy.spatial
import camera
import collections
import util

from numpy import *


# When creating the asterism database the celestial sphere is
# (quasi-efficiently) covered by circles of this radius. For each circle, the
# brightest star within the circle is indexed (ie. asterisms with it as the
# main star are added to the asterism database.)
#
# Indexing asterisms in this way means that any star in the source image that
# has no brighter neighbours within a radius of 2 * ASTERISM_SEARCH_RADIUS
# should have associated asterisms in the database.
ASTERISM_SEARCH_RADIUS = (3. * math.pi/180.)

# Radius around an asterism's main star to search for neighbours.
NEIGHBOUR_RADIUS = (3. * math.pi/180.)

# The brightest NUM_NEIGHBOURS stars within the NEIGHBOUR_RADIUS will be used
# to create asterisms (NUM_NEIGHBOURS choose 3 asterisms per main star).
NUM_NEIGHBOURS = 3

SCORE_THRESHOLD = 11

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
        coord_frame[:, 2] = main_star.vec
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

        # Compose the final 1-D descriptor for this asterism. Include the
        # asterism scale as the first element (we are working with a calibrated
        # camera so scale invariance only serves to increase our search space).
        # Normalise this element by the search radius so that it doesn't
        # dominate.
        self.vec = vstack([matrix([[dists[max_idx] / NEIGHBOUR_RADIUS]]),
                           hash_matrix.T.reshape(
                               (len(neighbours) * 2 - 2 ,1))])

    def __repr__(self):
        return "<Asterism(main_star=%s, neighbours=%s, vec=%s)>" % (
                repr(self.main_star), repr(self.neighbours), repr(self.vec))

def choose(l, n):
    if n == 0:
        yield []
    elif len(l) != 0:
        for c in choose(l[1:], n - 1):
            yield [l[0]] + c
        for c in choose(l[1:], n):
            yield c

def asterisms_for_star(main_star, star_db, num_neighbours=NUM_NEIGHBOURS):
    neighbour_stars = list(sorted((neighbour_star
                                for neighbour_star, dist
                                in star_db.search_vec(main_star.vec, NEIGHBOUR_RADIUS)
                                if neighbour_star != main_star),
                              key=lambda x: x.mag))[:num_neighbours]

    for neighbour_subset in choose(neighbour_stars, 3):
        ast = Asterism(main_star, neighbour_subset)
        yield ast

def asterisms_gen(star_db):
    uncovered = set(s for s in star_db if s.mag < 6.0)
    indexed = set()
    while uncovered:
        print "{} / {} uncovered. {} indexed.".format(
                                    len(uncovered), len(star_db), len(indexed))
        star = iter(uncovered).next()

        neighbour_stars = set(s for s, d in
                                      star_db.search_vec(
                                          star.vec,
                                          ASTERISM_SEARCH_RADIUS))
        brightest_neighbour = min(neighbour_stars, key=lambda x: x.mag)

        if brightest_neighbour not in indexed:
            for ast in asterisms_for_star(brightest_neighbour, star_db):
                yield ast
            indexed.add(brightest_neighbour)
        uncovered -= neighbour_stars


class AsterismDatabase(object):
    def __init__(self, asterism_iterable):
        self.asterisms = list(asterism_iterable)
        self.asterism_dict = collections.defaultdict(set)
        for asterism in self.asterisms:
            self.asterism_dict[asterism.main_star.id].add(asterism)
        self.tree = scipy.spatial.KDTree(vstack([ast.vec.T for ast in self.asterisms]))

    def search(self, query_ast):
        dist, idx = self.tree.query(query_ast.vec.flat)
        return self.asterisms[idx], dist

#@@@ WIP re-implementation of align_image().
def align_image2(image_stars, ast_db):
    image_star_db = stardb.StarDatabase(image_stars)

    # Locate stars in the image that are the brightest star within a radius of
    # 2 * ASTERISM_SEARCH_RADIUS.  Such stars should have asterisms in the
    # database.
    for image_star in itertools.islice(
            sorted(image_star_db, key=lambda s: s.mag),
            0, 50):

        brightest_star, _ = max(
            image_star_db.search_vec(image_star.vec,
                                     2 * ASTERISM_SEARCH_RADIUS),
            key=(lambda (s, d): s.flux))

        if brightest_star == image_star:
            matches = []
            
            for query_ast in asterisms_for_star(image_star, image_star_db,
                                                num_neighbours=6):
                found_ast, dist = ast_db.search(query_ast)
                matches.append((dist, found_ast.main_star))

            print image_star
            for dist, star in sorted(matches,
                                      key=(lambda (d,s): -d)):
                print "{}: {}".format(dist, star)
            import pdb; pdb.set_trace()
            pass


def align_image(image_stars, ast_db):
    image_star_db = stardb.StarDatabase(image_stars)

    best_scores = []
    for image_star in itertools.islice(
            sorted(image_star_db, key=lambda s: s.mag),
            0, 50):
        print "Trying {}".format(image_star)

        scores = collections.defaultdict(int)
        
        for query_ast in asterisms_for_star(image_star, image_star_db,
                                            num_neighbours=4):
            closest = ast_db.search(query_ast)[0].main_star
            scores[closest] += 1

        import pdb; pdb.set_trace()

        if scores:
            best_star, score = max(scores.iteritems(), key=(lambda x: x[1]))
            best_scores.append((score, image_star, best_star))
            
    for score, image_star, best_star in sorted(best_scores):
        print "Best match for %s: %s (score %s)" % (image_star.coords, best_star.id, score)

    best_scores = [x for x in best_scores if x[0] >= SCORE_THRESHOLD]
    if len(best_scores) == 0:
        raise CouldNotAlignError()

    import pdb; pdb.set_trace()

    camera_points = hstack([image_star.vec for score, image_star, best_star in best_scores])
    world_points = hstack([best_star.vec for score, image_star, best_star in best_scores])

    R, T = util.orientation_from_correspondences(camera_points, world_points)

    return stardb.vec_to_angles(R[:, 2]) + (R,)

def draw_stars(star_db, image, R, cam_model, mag_limit=4.0):
    """
    Draw stars on an image. The camera is assumed to be orientated at R.
    """

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

