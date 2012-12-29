#!/usr/bin/python

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

NUM_NEIGHBOURS = 10
NEIGHBOUR_RADIUS = (3. * math.pi/180.)
SCORE_THRESHOLD = 20

class Asterism(object):
    def __init__(self, main_star, neighbours):
        assert len(neighbours) == 3

        self.main_star = main_star
        self.neighbours = neighbours
        
        diffs = [neigh.vec - main_star.vec for neigh in neighbours]
        dists = [linalg.norm(d) for d in diffs]

        max_idx = max(enumerate(dists), key=(lambda t: t[1]))[0]
        other_idx = range(0,max_idx) + range(max_idx+1,3)

        coord_frame = matrix(zeros((3,3)))
        coord_frame[:, 2] = main_star.vec
        coord_frame[:, 0] = cross(diffs[max_idx].T, coord_frame[:, 2].T).T
        coord_frame[:, 1] = cross(coord_frame[:,0].T, coord_frame[:,2].T).T

        for i in range(3):
            coord_frame[:, i] /= linalg.norm(coord_frame[:, i])

        coord_frame *= dists[max_idx]

        hash_matrix = coord_frame.I * hstack([diffs[other_idx[0]], diffs[other_idx[1]]])
        hash_matrix = hash_matrix[:2, :2]

        if hash_matrix[0, 0] > hash_matrix[0, 1]:
            hash_matrix = fliplr(hash_matrix)

        self.vec = vstack([matrix([[dists[max_idx] / NEIGHBOUR_RADIUS]]),
                           hash_matrix.T.reshape((4,1))])

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

def asterisms_for_star(main_star, star_db):
    neighbour_stars = list(sorted((neighbour_star
                                for neighbour_star, dist
                                in star_db.search_vec(main_star.vec, NEIGHBOUR_RADIUS)
                                if neighbour_star != main_star),
                              key=lambda x: x.mag))[:NUM_NEIGHBOURS]

    for neighbour_pair in choose(neighbour_stars, 3):
        ast = Asterism(main_star, neighbour_pair)
        yield ast

def asterisms_gen(star_db, main_max_mag=4.0):
    for main_star in star_db:
        if main_star.mag < main_max_mag:
            for ast in asterisms_for_star(main_star, star_db):
                yield ast

class AsterismDatabase(object):
    def __init__(self, asterism_iterable):
        self.asterisms = list(asterism_iterable)
        self.tree = scipy.spatial.KDTree(vstack([ast.vec.T for ast in self.asterisms]))

    def search(self, query_ast):
        dist, idx = self.tree.query(query_ast.vec.flat)
        return self.asterisms[idx], dist

def align_image(axy_file, cam_model, ast_db):
    image_star_db = stardb.StarDatabase(stardb.xy_list_star_gen(axy_file, cam_model))

    best_scores = []
    for image_star in itertools.islice(
            sorted(image_star_db, key=lambda s: s.mag),
            0, 50):

        scores = collections.defaultdict(int)

        for query_ast in asterisms_for_star(image_star, image_star_db):
            closest = ast_db.search(query_ast)[0].main_star
            scores[closest] += 1

        best_star, score = max(scores.iteritems(), key=(lambda x: x[1]))

        best_scores.append((score, image_star, best_star))
            
    for score, image_star, best_star in sorted(best_scores):
        print "Best match for %s: %s (score %s)" % (image_star.coords, best_star.id, score)

    best_scores = [x for x in best_scores if x[0] > SCORE_THRESHOLD]

    camera_points = hstack([image_star.vec for score, image_star, best_star in best_scores])
    world_points = hstack([best_star.vec for score, image_star, best_star in best_scores])

    R, T = util.orientation_from_correspondences(camera_points, world_points)

    return stardb.vec_to_angles(R[:, 2])


if __name__ == "__main__":
    print "%f: Building star database..." % time.clock()
    star_db = stardb.StarDatabase(stardb.hip_star_gen('data/hip_main.dat'))
    print "%f: Building asterism database..." % time.clock()
    ast_db = AsterismDatabase(asterisms_gen(star_db))
    print "%f: Aligning image" % time.clock()

    cam_model = camera.BarrelDistortionCameraModel(
            3080.1049050112761 * 1024./3888.,
            1.57e-08 * (3888./1024.)**2,
            683.,
            1024.)
    ra, dec = align_image(sys.argv[1], cam_model, ast_db)
    print "RA: %s, Dec: %s" % (stardb.ra_to_str(ra), stardb.dec_to_str(dec))
    print "%f: Done" % time.clock()

