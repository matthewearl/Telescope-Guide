import operator
import cv
import math
import numpy
import scipy.linalg

__all__ = ['draw_points', 'get_circle_pattern']

def matrix_rotate_x(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return numpy.matrix([[1.0, 0.0, 0.0, 0.0],
                         [0.0,   c,  -s, 0.0],
                         [0.0,   s,   c, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

def matrix_rotate_y(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return numpy.matrix([[  c, 0.0,   s, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [ -s, 0.0,   c, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

def matrix_rotate_z(theta):
    s = math.sin(theta)
    c = math.cos(theta)
    return numpy.matrix([[ c,   -s, 0.0, 0.0],
                         [ s,    c, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

def matrix_trans(x, y, z):
    return numpy.matrix([[1.0, 0.0, 0.0,   x],
                         [0.0, 1.0, 0.0,   y],
                         [0.0, 0.0, 1.0,   z],
                         [0.0, 0.0, 0.0, 1.0]])

def matrix_invert(m):
    return vstack([hstack([m[:3, :3].T, -m[:3, 3:4]]), m[3:4, :]])

def left_inverse(A):
    return (A.T * A).I * A.T

def right_inverse(A):
    return A.T * (A * A.T).I

def draw_points(image, points, color=(255, 255, 0),
                circle_color=(0, 255, 0)):
    """
    Draw a set of points on an image.

    image: Image to draw points on.
    points: Dict of labels to 1x2 matrices representing pixel coordinates.
    """
    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 0.2, 0.2, 0, 3, 8)
    for name, point in points.iteritems():
        point = (int(point[0, 0] + image.width/2),
                 int(image.height/2 - point[0, 1]))
        cv.PutText(image, str(name), point, font, cv.CV_RGB(*color))
        cv.Circle(image, point, 5, cv.CV_RGB(*circle_color))


def get_circle_pattern(roll_radius=None):
    out = dict(("%d%s" % (x + 8 * y, l), (25.0 * x, -85.0 * y - (0 if l == 'a' else 50.0), 0.0)) for y in range(3) for x in range(8) for l in ('a', 'b'))

    if roll_radius != None:
        out = dict((key, (roll_radius * math.sin(x/roll_radius), y, -roll_radius * math.cos(x/roll_radius))) for (key, (x, y, z)) in out.iteritems())

    return out

def col_slice(M, cols):
    return numpy.hstack([M[:, i:(i+1)] for i in cols])

def orientation_from_correspondences(points1, points2):
    """
    Find a rotation R and offset T such that:

    sum ||R*p1,i + T - p2,i||^2

    is minimized.
    """
    assert points1.shape[1] == points2.shape[1]

    points1 = numpy.matrix(points1.T)
    points2 = numpy.matrix(points2.T)

    def centroid(points):
        return reduce(operator.add, points) / float(points.shape[0])

    def sub_all(points, r):
        return numpy.vstack(row - r for row in points)

    c1 = centroid(points1)
    c2 = centroid(points2)

    points1 = sub_all(points1, c1)
    points2 = sub_all(points2, c2)

    H = reduce(operator.add, (points1[i,:].T * points2[i,:] for i in xrange(points1.shape[0])))
    U, S, Vt = numpy.linalg.svd(H)
    R = Vt.T * U.T

    return R, (c2.T - R * c1.T)

def test_orientation_from_correspondences(num_points=5):
    R = numpy.matrix(scipy.linalg.qr(numpy.random.random((3,3)))[0])
    T = numpy.matrix(numpy.random.random((3,1))) * 20.0

    points1 = numpy.matrix(numpy.random.random((3, num_points))) * 10.0

    points2 = R * points1
    points2 = numpy.vstack(col + T.T for col in points2.T).T

    R_recovered, T_recovered = orientation_from_correspondences(points1, points2)

    print "R: %s" % R
    print "T: %s" % T
    print "R recovered: %s" % R_recovered
    print "T recovered: %s" % T_recovered

def argmax(it):
    max_idx = -1

    for idx, val in enumerate(it):
        if max_idx == -1 or val > max_val:
            max_idx = idx
            max_val = val

    return max_idx

def argmin(it):
    return argmax(-x for x in it)
