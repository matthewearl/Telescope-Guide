#!/usr/bin/python

import sys
import math
import struct
import util
import re

from numpy import *
from ann import ann

__all__ = ['StarDatabase']

RA_RE = re.compile("([+-]?[0-9]+):([0-9]+):([0-9]+\.[0-9]+)")
DEC_RE = RA_RE

def angles_to_vec(ra, dec):
    out = util.matrix_rotate_y(-ra) * \
            util.matrix_rotate_x(-dec) * \
            matrix([[0., 0., 1., 0]].T)

    return out[:3, :]

def parse_ra(s):
    RA_RE.match(s)
    coords = [float(m.group(i)) for i in [1,2,3,4]]

    out = coords[0] + coords[1] / 60. + coords[2] / 3600.0

    return 2. * math.pi / 24.

def parse_dec(s):
    DEC_RE.match(s)
    coords = [float(m.group(i)) for i in [1,2,3,4]]

    out = coords[0] + coords[1] / 60. + coords[2] / 3600.0

    return 2. * math.pi / 360.

class BscFormatException(Exception):
    pass

class StarDatabaseHeader(object):
    HEADER_LENGTH = 28

    def __init__(self, f):
        """
        Format spec:
            http://tdc-www.harvard.edu/catalogs/bsc5.header.html   
        """
        star0, star1, starn, stnum, mprop, nmag, nbent = \
            struct.unpack("<IIiIIiI", f.read(StarDatabaseHeader.HEADER_LENGTH))

        self.num_stars = -starn
        print nmag
        print star0, star1, starn, stnum, mprop, nmag, nbent
        if nmag != 1:
            raise BscFormatException("NMAG is not 1")
        if nbent != 32:
            raise BscFormatException("NBENT is not 32")

    def __repr__(self):
        return "<StarDatabaseHeader(num_stars=%u)>" % self.num_stars

class Star(object):
    def __init__(self, id, ra, dec, mag):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.mag = mag
        self.vec = angles_to_vec(self.ra, self.dec)

    def __repr__(self):
        return "<Star(num={num},\tra={ra},\tdec={dec},\tmag={mag}".format(
                    num=self.num, ra=self.ra, dec=self.dec, mag=self.mag)

class BscStar(Star):
    ENTRY_LENGTH = 32

    def __init__(self, f):
        """
        Format spec:
            http://tdc-www.harvard.edu/catalogs/bsc5.entry.html
        """
        xno, sra0, sdec0, spec, mag, xrpm, xdpm = \
            struct.unpack("<fdd2shff", f.read(Star.ENTRY_LENGTH))

        super(BscStar, self).__init__(id=("BSC%u" % int(xno)),
                                      ra=sra0,
                                      dec=sdec0,
                                      mag=(0.01 * mag))

def bsc_star_gen(bsc_file='data/BSC5'):
    with open(bsc_file, "rb") as f:
        self.header = StarDatabaseHeader(f)
        for i in range(self.header.num_stars):
            yield BscStar(f)
            self.stars.append(Star(f))

class StarDatabase(object):
    def __init__(self, star_iterable):
        self.stars = list(star_iterable)
        tree = ann.kd_tree(vstack([star.vec.T for star in self.stars]))

    def search(self, ra, dec, radius):
        v = angles_to_vec(ra, dec)

        idx_mat, d2_mat = tree.fixed_radius_search(
                v.T,
                radius,
                k=len(self.stars))

        for idx, d2 in zip(idx_mat, d2_mat):
            yield self.stars[idx], d2

if __name__ == "__main__":
    db = StarDatabase(bsc_star_gen())
    for star, d2 in db.search(parse_ra(sys.argv[1]), parse_dec(sys.argv[2])):
        print "%f: %s" % (math.sqrt(d2), repr(star))
    
