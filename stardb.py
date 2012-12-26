#!/usr/bin/python

import math
import struct
import util

from numpy import *

__all__ = ['StarDatabase']


def angles_to_vec(ra, dec):
    out = util.matrix_rotate_y(-ra) * \
            util.matrix_rotate_x(-dec) * \
            matrix([[0., 0., 1., 0]].T)

    return out[:3, :]

def parse_ra(s):
    pass

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
    ENTRY_LENGTH = 32

    def __init__(self, f):
        """
        Format spec:
            http://tdc-www.harvard.edu/catalogs/bsc5.entry.html
        """
        xno, sra0, sdec0, spec, mag, xrpm, xdpm = \
            struct.unpack("<fdd2shff", f.read(Star.ENTRY_LENGTH))

        self.num = int(xno)
        self.ra = sra0 # radians
        self.dec = sdec0 # radians
        self.mag = 0.01 * mag

        self.vec = angles_to_vec(self.ra, self.dec)

    def __repr__(self):
        return "<Star(num={num},\tra={ra},\tdec={dec},\tmag={mag}".format(
                    num=self.num, ra=self.ra, dec=self.dec, mag=self.mag)


class StarDatabase(object):
    def __init__(self, bsc_file='data/BSC5'):
        with open(bsc_file, "rb") as f:
            self.header = StarDatabaseHeader(f)
            for i in range(self.header.num_stars):
                self.add_star(Star(f))

    def add_star(self, star):
        print repr(star)

if __name__ == "__main__":
    print repr(StarDatabase().header)
    
