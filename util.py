import operator
import math

import camera

import matplotlib.pyplot as plt
import numpy
import scipy.linalg


__all__ = (
    'argmax',
    'argmin',
    'col_slice',
    'draw_points',
    'get_circle_pattern',
    'left_inverse',
    'matrix_invert',
    'matrix_rotate_x',
    'matrix_rotate_y',
    'matrix_rotate_z',
    'matrix_trans',
    'orientation_from_correspondences',
    'right_inverse',
    'test_orientation_from_correspondences',
)


WINDOWS = {
    "orion": ((1.465, 0.), 30 * math.pi / 180),
    "orion_nebula": ((1.465, -0.1), 3. * math.pi / 180),
    "andromeda": ((0.18631389765039466, 0.7202392046563266), 5 * math.pi / 180),
    "all": (None, None),
}


def cam_from_window(window, image_width, image_height):
    (ra, dec), radius = window

    cam_matrix = (matrix_rotate_y(-ra) * matrix_rotate_x(-dec))[:3, :3]
    pixel_scale = image_width / (2 * math.tan(radius))
    cam = camera.BarrelDistortionCameraModel(pixel_scale,
                                             0.,
                                             image_width,
                                             image_height)

    return cam, cam_matrix


def set_plot_size(width, height):
    fig = plt.figure()
    fig.set_figwidth(width)
    fig.set_figheight(height)


def plot_image(im):
    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()
    plt.imshow(-im.astype(numpy.float).mean(axis=2),
               cmap='gray')


def plot_image_star_labels(im_stars, labels):
    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()
    if labels is None:
        labels = [s.id for s in im_stars]
    for im_star, label in zip(im_stars, labels):
        x, y = im_star.coords
        plt.text(x, y, label)


def plot_image_stars(im_stars, size=1.0, color='blue', alpha=1.0):
    """Plot image stars.

    `size` is the size that a magnitude 0 star will be plotted. The area of a
        star is proportional to its brightness.

    """
    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()

    M = numpy.array([[s.coords[0], s.coords[1], s.mag] for s in im_stars]).T

    s = size * numpy.sqrt(2.512 ** (-M[2]))

    plt.scatter(M[0], M[1], s=s, color=color, alpha=alpha)


def plot_image_asterisms(im_asts, color='k', line_width=1.):
    """Plot asterisms from an image.

    """
    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()
    for im_ast in im_asts:
        for i in range(3):
            plt.plot([im_ast.main_star.coords[0],
                      im_ast.neighbours[i].coords[0]],
                     [im_ast.main_star.coords[1],
                      im_ast.neighbours[i].coords[1]],
                     '{}-'.format(color),
                     linewidth=line_width)


def _star_to_pixel(star, cam, cam_matrix):
    return cam.world_vec_to_pixel(star.vec, cam_matrix)


def plot_star_labels(stars, cam, cam_matrix, labels=None):
    if labels is None:
        labels = [s.id for s in stars]
    for star, label in zip(stars, labels):
        x, y = _star_to_pixel(star, cam, cam_matrix)
        if (0 <= x <= cam.image_width and
            0 <= y <= cam.image_height):
            plt.text(x, y, label)


def plot_stars(star_db, cam, cam_matrix,
               color='blue', size_scale=.1, alpha=1.0):
    """Plot stars according to a camera model and orientation.

    `cam` is a `camera.CameraModel`.
    `cam_matrix` is the camera orientation matrix.
    `size_scale` is the size in degrees of a magnitude 0 star. The area of a star is
        proportional to its brightness.

    """
    size = (size_scale / math.atan(1 / cam.pixel_scale))
    plot_image_stars(cam.generate_image_stars(star_db, cam_matrix),
                     size=size, color=color, alpha=alpha)


def plot_asterisms(asts, cam, cam_matrix, color='k', line_width=1.):
    """Plot asterisms according to a camera model and orientation.

    `cam` is a `camera.CameraModel`.
    `cam_matrix` is the camera orientation matrix.
    `line_width` is the with of the lines used to draw the asterisms.

    """
    if not plt.gca().yaxis_inverted():
        plt.gca().invert_yaxis()

    for ast in asts:
        for i in range(3):
            M = numpy.array([_star_to_pixel(ast.main_star, cam, cam_matrix),
                             _star_to_pixel(ast.neighbours[i],
                                            cam,
                                            cam_matrix)]).T

            if (numpy.max(M[0]) <= cam.image_width and
                numpy.min(M[0]) >= 0 and
                numpy.max(M[1]) <= cam.image_height and
                numpy.min(M[1]) >= 0):

                plt.plot(M[0], M[1], '{}-'.format(color),
                         linewidth=line_width)


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
    import cv
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

