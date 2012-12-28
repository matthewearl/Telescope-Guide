#!/usr/bin/python

import math
import time
import stardb

from numpy import *
from ann import ann

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
        print self.vec

def choose(l, n):
    if n == 0:
        yield []
    elif len(l) != 0:
        for c in choose(l[1:], n - 1):
            yield [l[0]] + c
        for c in choose(l[1:], n):
            yield c

def asterisms_for_star(main_star, star_db):
    neighbour_stars = [s for m, s in sorted((neighbour_star.mag, neighbour_star)
            for neighbour_star, dist
            in star_db.search_vec(main_star.vec, NEIGHBOUR_RADIUS)
            if neighbour_star != main_star)[:NUM_NEIGHBOURS]]

    for neighbour_pair in choose(neighbour_stars, 2):
        yield Asterism(main_star, neighbour_pair)

def asterisms_gen(star_db, main_max_mag=5.0):
    for main_star in star_db:
        if main_star.mag < main_max_mag:
            return asterisms_for_star(main_star, star_db)

    return []

class AsterismDatabase(object):
    def __init__(self, asterism_iterable):
        self.asterisms = list(asterism_iterable)
        d = vstack([ast.vec.T for ast in self.asterisms])
        print repr(d)
        self.tree = ann.kd_tree(d, copy=False)

    def search(self, query_ast):
        idx_mat, d2_mat = self.tree.search(quemry_ast.vec)
        assert idx_mat.shape[0] == 1
        return self.asterisms[idx_mat[0, 0]], math.sqrt(d2_mat[0, 0])

if __name__ == "__main__":
    print "%f: Building star database..." % time.clock()
    star_db = stardb.StarDatabase(stardb.hip_star_gen('data/hip_main.dat'))
    print "%f: Building asterism database..." % time.clock()
    ast_db = AsterismDatabase(asterisms_gen(star_db))
    print "%f: Done" % time.clock()
