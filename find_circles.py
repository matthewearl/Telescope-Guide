#!/usr/bin/python

import getopt
import random
import sys
import cv

__all__ = ['find_concentric_circles', 'find_circles']

def dist_sqr(c1, c2):
    return (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2

class SpacialHash():
    def __init__(self, box_size=5):
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

    def clusters(self, radius, min_cluster_size=1):
        out = {}
        already_clustered = set()

        for obj, coords in self:
            cluster = []
            for neighbour, neighbour_coords in self.search(coords, radius):
                if neighbour not in already_clustered:
                    cluster.append((neighbour, neighbour_coords))
                    already_clustered.add(neighbour)
            if len(cluster) >= min_cluster_size:
                yield cluster

def random_color():
    return cv.CV_RGB(random.uniform(0,255),
                     random.uniform(0,255),
                     random.uniform(0,255))

def find_concentric_circles(image_in):
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

    while contours:
       color = random_color()
       moments = cv.Moments(contours)
       if moments.m00 > 0.0:
           coords = (moments.m10 / moments.m00, moments.m01 / moments.m00)
           spacialHash.add(contourNum, coords)
           contourNum += 1
       contours = contours.h_next()

    for cluster in spacialHash.clusters(1, min_cluster_size=4):
        yield tuple(sum(x)/len(x) for x in zip(*[coords for obj, coords in cluster]))

def read_bar_code(image, p1, p2, num_bars=20, color_image=None):
    xvals = [p1[0] + (i + 0.5) * (p2[0] - p1[0])/num_bars for i in xrange(num_bars)]
    yvals = [p1[1] + (i + 0.5) * (p2[1] - p1[1])/num_bars for i in xrange(num_bars)]

    bars = [image[y, x] < 128.0 for x, y in zip(xvals, yvals)]

    if bars[:6] != [False, True] * 3:
        return None
    if bars[-6:] != [False, False] + [True, False] * 2:
        return None
    if color_image:
        for x, y in zip(xvals, yvals):
            cv.Circle(color_image, (int(x), int(y)), 2, cv.CV_RGB(0, 255, 255))
    return sum(v * 2**(7 - i) for i, v in enumerate(bars[6:-6]))


def read_barcodes(image, circles):
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
            bc = read_bar_code(image, c1, c2)
            if bc != None and bc in valid_numbers:
                out["%da" % bc] = c1
                out["%db" % bc] = c2
                    
    return out


def find_labelled_circles(image_in, thresh_file_name=None, annotate_image=None):
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

    circles = list(find_concentric_circles(image))

    if annotate_image:
        font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
        for i, coords in enumerate(circles):
           coords = tuple(map(int, coords))
           cv.Circle(annotate_image, coords, 5, cv.CV_RGB(0, 255, 0))
           cv.Circle(annotate_image, coords, 7, cv.CV_RGB(255, 255, 255))

    pairs = read_barcodes(image, circles)
    if annotate_image:
        for name, circle in pairs.iteritems():
            cv.PutText(annotate_image, name, tuple(map(int, circle)), font, cv.CV_RGB(255, 0, 255))

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
    print find_labelled_circles(image, thresh_file_name, color_image)

    cv.SaveImage(out_file_name, color_image)

