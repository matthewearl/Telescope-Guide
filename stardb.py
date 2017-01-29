#!/usr/bin/python

import pyfits
import os
import sys
import math
import struct
import util
import re
import scipy.spatial
import cPickle as pickle

from numpy import *

__all__ = ['StarDatabase', 'hip_star_gen', 'bsc_star_gen', 'xy_list_star_gen', 'HipStar']

RA_RE = re.compile("([+-]?)([0-9]+)[: ]([0-9]+)[: ]([0-9]+\.[0-9]+)")
DEC_RE = RA_RE

class InvalidFormatException(Exception):
    pass

def angles_to_vec(ra, dec):
    out = util.matrix_rotate_y(-ra) * \
            util.matrix_rotate_x(-dec) * \
            matrix([[0., 0., 1., 0]]).T

    return out[:3, :]

def vec_to_angles(vec):
    ra = -math.atan2(vec[0, 0], vec[2, 0])
    ra = ra % (2. * math.pi)
    vec = util.matrix_rotate_y(ra)[:3, :3] * vec
    dec = math.atan2(vec[1, 0], vec[2, 0])

    return ra, dec

def parse_ra(s):
    m = RA_RE.match(s)
    if not m:
        raise InvalidFormatException()
    coords = [float(m.group(i)) for i in [2,3,4]]
    
    out = coords[0] + coords[1] / 60. + coords[2] / 3600.0

    if m.group(1) == '-':
        out = -out

    return out * 2. * math.pi / 24.

def parse_dec(s):
    m = DEC_RE.match(s)
    if not m:
        raise InvalidFormatException()
    coords = [float(m.group(i)) for i in [2,3,4]]

    out = coords[0] + coords[1] / 60. + coords[2] / 3600.0

    if m.group(1) == '-':
        out = -out

    return out * 2. * math.pi / 360.

def ra_to_str(ra):
    secs = ra * 60. * 60. * 12. / math.pi

    mins = secs // 60.
    secs = math.fmod(secs, 60.)

    hrs = mins // 60
    mins = math.fmod(mins, 60.)

    return "%u:%02u:%02.3f" % (hrs, mins, secs)

def dec_to_str(dec):
    if dec < 0.0:
        sign = "-"
        dec = -dec
    else:
        sign = "+"

    secs = dec * 60. * 60. * 180. / math.pi

    mins = secs // 60.
    secs = math.fmod(secs, 60.)

    degs = mins // 60
    mins = math.fmod(mins, 60.)

    return "%s%u:%02u:%02.3f" % (sign, degs, mins, secs)

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
        if nmag != 1:
            raise BscFormatException("NMAG is not 1")
        if nbent != 32:
            raise BscFormatException("NBENT is not 32")

    def __repr__(self):
        return "<StarDatabaseHeader(num_stars=%u)>" % self.num_stars

class Star(object):
    def __init__(self, id, vec, mag):
        self.id = id
        self.vec = vec
        self.mag = mag

    def __repr__(self):
        return "<Star(id={id}, vec={vec}, mag={mag})>".format(
                    id=self.id, vec=self.vec, mag=self.mag)

    def __str__(self):
        return "Mag: %f, Id: %s, Vec: %s" % \
                    (self.mag, self.id, self.vec)

class EquatorialStar(Star):
    def __init__(self, id, ra, dec, mag):
        self.id = id
        self.ra = ra
        self.dec = dec
        self.mag = mag
        self.vec = angles_to_vec(self.ra, self.dec)

    def __repr__(self):
        return "<EquatorialStar(id={id},\tra={ra},\tdec={dec},\tmag={mag})>".format(
                    id=self.id, ra=self.ra, dec=self.dec, mag=self.mag)

    def __str__(self):
        return "Mag: %f, Id: %s, RA: %s, DE: %s" % \
                (self.mag, self.id, ra_to_str(self.ra), dec_to_str(self.dec))

class BscStar(EquatorialStar):
    ENTRY_LENGTH = 32

    def __init__(self, f):
        """
        Format spec:
            http://tdc-www.harvard.edu/catalogs/bsc5.entry.html
        """
        xno, sra0, sdec0, spec, mag, xrpm, xdpm = \
            struct.unpack("<fdd2shff", f.read(BscStar.ENTRY_LENGTH))

        super(BscStar, self).__init__(id=("BSC%u" % int(xno)),
                                      ra=sra0,
                                      dec=sdec0,
                                      mag=(0.01 * mag))

class HipStar(EquatorialStar):
    def __init__(self, line):
        """
        Format spec:
            ftp://cdsarc.u-strasbg.fr/pub/cats/I%2F239/ReadMe
        """

        if not line[41:46].strip():
            raise InvalidFormatException()

        prefix = "HIP" if (line[0] == 'H') else "TYC"
        id = "{} {}".format(prefix, line[2:14].strip())
        super(HipStar, self).__init__(id=id,
                                      ra=parse_ra(line[17:28]),
                                      dec=parse_dec(line[29:40]),
                                      mag=float(line[41:46]))

class ImageStar(Star):
    def __init__(self, id, coords, cam_model, flux):
        self.coords = coords
        self.cam_model = cam_model
        self.flux = flux
        super(ImageStar, self).__init__(id=id,
                                        vec=cam_model.pixel_to_vec(*coords),
                                        mag=(-flux))

    def __repr__(self):
        return "<ImageStar(id=%s, coords=%s, cam_model=%s, flux=%s, vec=%s)>" % \
                (self.id, self.coords, self.cam_model, self.flux, repr(self.vec))

    def __str__(self):
        return "Flux: %f, Id: %s, Coords: %s" % (self.flux, self.id, self.coords)


def bsc_star_gen(bsc_file='data/BSC5'):
    with open(bsc_file, "rb") as f:
        header = StarDatabaseHeader(f)
        for i in range(header.num_stars):
            yield BscStar(f)

def hip_star_gen(dat_file='data/hip_main.dat', mag_limit=9.5, use_cache=True):
    """Load stars from the Hipparcos/Tycho catalogs.

    The input files must be in the following format:
        ftp://cdsarc.u-strasbg.fr/pub/cats/I%2F239/ReadMe

    """
    cache_file_name = "%s-%.2f.pickle" % (dat_file, mag_limit)

    if use_cache and os.path.exists(cache_file_name):
        with open(cache_file_name, "r") as cache_file:
            stars = pickle.load(cache_file)
    else:
        stars = []
        with open(dat_file, "r") as f:
            for line in f.readlines():
                try:
                    star = HipStar(line)
                    if star.mag < mag_limit:
                        stars.append(star)
                except InvalidFormatException:
                    pass

        with open(cache_file_name, "w") as cache_file:
            pickle.dump(stars, cache_file)

    return stars

def xy_list_star_gen(axy_file, cam_model):
    with pyfits.open(axy_file) as hdulist:
        for i, (x, y, flux, bg) in enumerate(hdulist['SOURCES'].data):
            yield ImageStar("%s%u" % (axy_file, i), (x, y), cam_model, flux)

def cat_star_gen(cat_file, cam_model):
    """
    Generate ImageStars from a sextractor .cat file.

    """

    with open(cat_file) as f:
        for i, line in enumerate(f.readlines()):
            if not line.startswith("#"):
                m = re.match(r"([-.\d]+)\s+([-.\d]+)\s+([-.\d]+)", line)
                yield ImageStar("%s%u" % (cat_file, i),
                                (float(m.group(2)), float(m.group(3))),
                                cam_model, -float(m.group(1)))
                          
class StarDatabase(object):
    def __init__(self, star_iterable):
        self.stars = list(star_iterable)
        self.tree = scipy.spatial.KDTree(vstack([star.vec.T for star in self.stars]))
        self.star_dict = dict((s.id, s) for s in self.stars)

    def search_vec(self, vec, radius):
        indices = self.tree.query_ball_point(vec.flat, radius)

        for idx in indices:
            star = self.stars[idx]
            d = linalg.norm(star.vec - vec)
            yield star, d

    def __iter__(self):
        return iter(self.stars)

    def search(self, ra, dec, radius):
        return self.search_vec(angles_to_vec(ra, dec),
                               radius)

    def __getitem__(self, key):
        return self.star_dict[key]

    def __len__(self):
        return len(self.stars)

def main():
    print "Loading database..."
    db = StarDatabase(hip_star_gen('data/hip_main.dat'))
    print "Searching..."
    ra = parse_ra(sys.argv[1])
    dec = parse_dec(sys.argv[2])
    radius = float(sys.argv[3]) * math.pi / 180.

    for star, d in db.search(ra, dec, radius):
        print "%f: %s" % (d * 180. / math.pi, star)

def _draw_image_stars(stars, image):
    for star in stars:
        coords = tuple(map(int, star.coords))
        cv.Circle(image, coords, 5, cv.CV_RGB(0, 255, 0))
    
    
if __name__ == "__main__":
    #import cProfile
    #cProfile.run('main()')
    #main()
    import camera
    import cv
    m = camera.BarrelDistortionCameraModel(1.0, 1.0, 100, 100)
    stars = list(cat_star_gen("/home/matt/sextractor-test/test.cat", m))
    stars = sorted(stars, key=(lambda x: -x.flux))[:1000]
    image = cv.LoadImage("/media/vbox_d_drive/Photos/IMG_1511.JPG", True)
    _draw_image_stars(stars, image)
    cv.SaveImage("out.png", image)

