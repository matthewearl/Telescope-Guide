#!/usr/bin/python

import itertools
import sys
import math
import time
import stardb
import scipy.spatial
import camera

from numpy import *

NUM_NEIGHBOURS = 10
NEIGHBOUR_RADIUS = (2. * math.pi/180.)

class Asterism(object):
    def __init__(self, main_star, neighbours):
        self.main_star = main_star
        self.neighbours = neighbours
        
        diffs = [neigh.vec - main_star.vec for neigh in neighbours]
        dists = [linalg.norm(d) for d in diffs]
        if dists[0] < dists[1]:
            dists = list(reversed(dists))
            diffs = list(reversed(diffs))

        self.vec = matrix([dists + [diffs[0].T * diffs[1]]]).T

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

    for neighbour_pair in choose(neighbour_stars, 2):
        yield Asterism(main_star, neighbour_pair)

def asterisms_gen(star_db, main_max_mag=5.0):
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

    for image_star in itertools.islice(
            sorted(image_star_db, key=lambda s: s.mag),
            0, 3):
        print "Matches for %s" % image_star

        for query_ast in asterisms_for_star(image_star, image_star_db):
            print "    %s" % (ast_db.search(query_ast),)

if __name__ == "__main__":
    print "%f: Building star database..." % time.clock()
    star_db = stardb.StarDatabase(stardb.hip_star_gen('data/hip_main.dat'))
    print "%f: Building asterism database..." % time.clock()
    ast_db = AsterismDatabase(asterisms_gen(star_db))
    print "%f: Aligning image" % time.clock()

    cam_model = camera.BarrelDistortionCameraModel(
            3080.1049050112761, 1.5762197792252771e-08)
    align_image(sys.argv[1], cam_model, ast_db)

