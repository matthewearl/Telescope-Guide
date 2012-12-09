#!/usr/bin/python

import getopt
import random
import sys
import cv
import math
import util

OUTER_SEARCH_RADIUS = 20
INNER_SEARCH_RADIUS = 5

__all__ = ['find_concentric_circles', 'find_circles']

def dist_sqr(c1, c2):
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

class SpacialHash():
    def __init__(self, box_size=20):
        self.box_size = box_size
        self.lookup = {}

    def _round_coords(self, coords):
        return tuple(map(lambda x: int(x) - int(x) % self.box_size, coords))

    def add(self, obj, coords):
        r_coords = self._round_coords(coords)
        self.lookup.setdefault(r_coords, {})[obj] = coords

    def delete(self, obj, coords):
        r_coords = _round_coords(coords)
        assert obj in self.lookup[r_coords]
        assert self.lookup[r_coords][obj] == coords

    def search(self, coords, radius):
        mins = self._round_coords((coords[0] - radius, coords[1] - radius))
        maxs = self._round_coords((coords[0] + radius + self.box_size,
                              coords[1] + radius + self.box_size))

        for y in xrange(mins[1], maxs[1], self.box_size):
            for x in xrange(mins[0], maxs[0], self.box_size):
                if (x, y) in self.lookup:
                    for obj, coords2 in self.lookup[(x, y)].iteritems():
                        if dist_sqr(coords, coords2) < radius**2:
                            yield (obj, coords2)

    def __iter__(self):
        for box in self.lookup.values():
            for obj, coords in box.iteritems():
                yield obj, coords

    def clusters(self, radius, min_cluster_size=1, max_cluster_size=None):
        out = {}
        already_clustered = set()

        for obj, coords in self:
            cluster = []
            for neighbour, neighbour_coords in self.search(coords, radius):
                if neighbour not in already_clustered:
                    cluster.append((neighbour, neighbour_coords))
                    if max_cluster_size != None and len(cluster) > max_cluster_size:
                        break
                    already_clustered.add(neighbour)
            if len(cluster) >= min_cluster_size and \
                (max_cluster_size == None or len(cluster) <= max_cluster_size):
                yield cluster

def random_color():
    return cv.CV_RGB(random.uniform(0,255),
                     random.uniform(0,255),
                     random.uniform(0,255))

class Feature(object):
    def draw(self, image):
        raise NotImplementedError()

    def get_centre(self):
        raise NotImplementedError()

class Ellipse(Feature):
    def __init__(self, moments):
        # Model the ellipse on the component with the largest area
        moments = list(moments)
        m = moments[util.argmax(m.m00 for m in moments)]

        self.centre = (m.m10 / m.m00, m.m01 / m.m00)
        self.angle = 0.5 * math.atan2(2. * m.mu11, m.mu20 - m.mu02)

        self.areas = [m.m00] + list(m.m00 for m in moments)

        def variance_at_angle(angle):
            s, c = math.sin(2.0 * angle), math.cos(2.0 * angle)
            nu11, nu02, nu20 = (x / m.m00 for x in (m.mu11, m.mu02, m.mu20))
            return 0.5 * (nu20 + nu02 + c * (nu20 - nu02) + 2. * s * nu11)

        # Find the axis lengths up to a constant of proportionality.
        self.axes = tuple(variance_at_angle(x)**0.5 for x in (self.angle, self.angle + 0.5 * math.pi))

        # Scale the axes so that the area of the resulting ellipse matches the measured area.
        scale_factor = math.sqrt(m.m00 / (math.pi * self.axes[0] * self.axes[1]))
        self.axes = tuple(scale_factor * x for x in self.axes)

    def draw(self, image):
       coords = tuple(map(int, self.centre))
       axes = tuple(map(int, self.axes))
       cv.Ellipse(image,
                  coords,
                  axes,
                  self.angle * 180. / math.pi,
                  0,
                  360.0,
                  cv.CV_RGB(0, 255, 0))

    def get_centre(self):
        return self.centre

class Point(Feature):
    def __init__(self, moments):
        def moment_to_point(m):
            return (m.m10 / m.m00, m.m01 / m.m00)
        self.point = tuple(sum(x)/len(x) for x in zip(*[moment_to_point(m) for m in moments]))

    def draw(self, image):
        coords = tuple(map(int, self.point))
        cv.Circle(image, coords, 5, cv.CV_RGB(0, 255, 0))
        cv.Circle(image, coords, 7, cv.CV_RGB(255, 255, 255))

    def get_centre(self):
        return self.point

def find_concentric_circles(image_in, find_ellipses=False):
    """
    Find concentric circles in an image. The concentric circles it finds are
    solid white circles surrounded by 3 rings: A black ring, a white ring, and
    a black ring.

    image_in: A binary image to search for concentric circles.

    Generates a pairs, each pair being the x,y coordinates of a concentric
    circle.
    """
    image = cv.CloneImage(image_in)
    contours = cv.FindContours(image, cv.CreateMemStorage(0))

    spacialHash = SpacialHash()
    contourNum = 1

    featureFactory = Ellipse if find_ellipses else Point

    while contours:
       color = random_color()
       moments = cv.Moments(contours)
       if moments.m00 > 0.0:
           centre = (moments.m10 / moments.m00, moments.m01 / moments.m00)
           spacialHash.add((contourNum, moments), centre)
           contourNum += 1
       contours = contours.h_next()

    for cluster in spacialHash.clusters(OUTER_SEARCH_RADIUS,
                                        min_cluster_size=4,
                                        max_cluster_size=4):
        if len(cluster) == 4:
            c1 = cluster[0][1]
            if all(dist_sqr(c1, c2) < INNER_SEARCH_RADIUS**2 for obj, c2 in cluster[1:]):
                yield featureFactory(m for (o, m), c in cluster)

def read_bar_code(image, p1, p2, num_bars=20, color_image=None):
    samples_per_bar = 10
    num_samples = samples_per_bar * num_bars
    xvals = [p1[0] + i * (p2[0] - p1[0])/num_samples for i in xrange(num_samples)]
    yvals = [p1[1] + i * (p2[1] - p1[1])/num_samples for i in xrange(num_samples)]

    samples = [image[y, x] < 128.0 for x, y in zip(xvals, yvals)]

    bars = []
    next_bar_middle = samples_per_bar // 2
    for i in xrange(1,num_samples):
        if i == next_bar_middle:
            bars += [samples[i]]
            next_bar_middle += samples_per_bar
        if samples[i - 1] != samples[i]:
            next_bar_middle = i + samples_per_bar // 2

    if len(bars) != num_bars:
        return None
    if bars[:6] != [False, True] * 3:
        return None
    if bars[-6:] != [False, False] + [True, False] * 2:
        return None
    if color_image:
        for x, y in zip(xvals, yvals):
            cv.Circle(color_image, (int(x), int(y)), 2, cv.CV_RGB(0, 255, 255))
    return sum(v * 2**(7 - i) for i, v in enumerate(bars[6:-6]))


def read_barcodes(image, circles, annotate_image=None):
    """
    Given a set of coordinates of concentric circles, attempt to find pairs
    with a 10-bit barcode between them.

    image: Binary image that contains the concentric circles and barcodes.
    circles: Coordinates of concentric circles in the image. The output of 
             find_concentric_circles() can be used here.

    Return a dict from numbers (the barcode's value) to a pair of circle
    coordinates (the coordinates of the CCs either side of the barcode).
    """
    valid_numbers = set(range(24))
    out = {}

    for i1, c1 in enumerate(circles):
        for i2, c2 in enumerate(circles):
            bc = read_bar_code(image, c1, c2, color_image=annotate_image)
            if bc != None and bc in valid_numbers:
                out["%da" % bc] = c1
                out["%db" % bc] = c2
                    
    return out


def find_labelled_circles(image_in, thresh_file_name=None, annotate_image=None, centre_origin=False, find_ellipses=False):
    """
    Find concentric circles in an image, which are identified with a barcode.

    image_in: Image to search.
    thresh_file_name: (Optional.) File to dump threshold image in.
    annotate_image: (Optional.) Image to annotate with intermediate circles and
                    output.
    """
    image = cv.CloneImage(image_in)
    cv.AdaptiveThreshold(image, image, 255.,
                         cv.CV_ADAPTIVE_THRESH_MEAN_C, 
                         cv.CV_THRESH_BINARY,
                         blockSize=31)
    if thresh_file_name:
        cv.SaveImage(thresh_file_name, image)

    features = list(find_concentric_circles(image, find_ellipses=find_ellipses))

    if annotate_image:
        for i, feature in enumerate(features):
            feature.draw(annotate_image)

    pairs = read_barcodes(image, [f.get_centre() for f in features], annotate_image)
    if annotate_image:
        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
        for name, circle in pairs.iteritems():
            cv.PutText(annotate_image, name, tuple(map(int, circle)), font, cv.CV_RGB(255, 0, 255))

    if centre_origin:
        pairs = dict((key, (x - 0.5*image_in.width, 0.5*image_in.height - y))
                        for key, (x, y) in pairs.iteritems())

    return pairs


if __name__ == "__main__":
    optlist, args = getopt.getopt(sys.argv[1:], 'i:o:t:')

    in_file_name = None
    out_file_name = None
    thresh_file_name = None
    for opt, param in optlist:
        if opt == "-i":
            in_file_name = param
        if opt == "-o":
            out_file_name = param
        if opt == "-t":
            thresh_file_name = param
        
    if not in_file_name or not out_file_name:
        raise Exception("Usage: %s -i <input image> -o <output image> [-t <output threshold image>]" % sys.argv[0])

    image = cv.LoadImage(in_file_name, False)
    color_image = cv.LoadImage(in_file_name, True)
    print find_labelled_circles(image, thresh_file_name, color_image, find_ellipses=True)

    cv.SaveImage(out_file_name, color_image)

