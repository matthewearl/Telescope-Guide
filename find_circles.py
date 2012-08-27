#!/usr/bin/python

import random
import sys
import cv

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
    if bars[-5:] != [False] + [True, False] * 2:
        return None
    if color_image:
        for x, y in zip(xvals, yvals):
            cv.Circle(color_image, (int(x), int(y)), 2, cv.CV_RGB(0, 255, 255))
    return sum(v * 2**(8 - i) for i, v in enumerate(bars[6:-5]))


def circles_to_pairs(image, circles):
    valid_numbers = set([0,2,3,4,5,6,7])
    out = {}

    for i1, c1 in enumerate(circles):
        for i2, c2 in enumerate(circles):
            bc = read_bar_code(image, c1, c2)
            if bc != None and bc in valid_numbers:
                out[bc] = (c1, c2)
                    
    return out


if __name__=="__main__":
    if len(sys.argv) < 2:
        raise Exception("Please provide an image as argument")

    image = cv.LoadImage(sys.argv[1], False)
    color_image = cv.LoadImage(sys.argv[1], True)
    cv.AdaptiveThreshold(image, image, 255.,
                         cv.CV_ADAPTIVE_THRESH_MEAN_C, 
                         cv.CV_THRESH_BINARY,
                         blockSize=21)
    cv.SaveImage('threshold.png', image)

    circles = list(find_concentric_circles(image))

    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
    for i, coords in enumerate(circles):
       coords = tuple(map(int, coords))
       cv.Circle(color_image, coords, 5, cv.CV_RGB(0, 255, 0))
       cv.Circle(color_image, coords, 7, cv.CV_RGB(255, 255, 255))

    pairs = circles_to_pairs(image, circles)
    for num, (c1, c2) in pairs.iteritems():
        cv.PutText(color_image, "%da" % num, tuple(map(int, c1)), font, cv.CV_RGB(255, 0, 255))
        cv.PutText(color_image, "%db" % num, tuple(map(int, c2)), font, cv.CV_RGB(255, 0, 255))

    cv.SaveImage('out.png', color_image)

