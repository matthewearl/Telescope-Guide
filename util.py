import cv

__all__ = ['draw_points']

def draw_points(image, points):
    """
    Draw a set of points on an image.

    image: Image to draw points on.
    points: Dict of labels to 1x2 matrices representing pixel coordinates.
    """
    font = cv.InitFont(cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 3, 8)
    for name, point in points.iteritems():
        point = (int(point[0, 0] + image.width/2),
                 int(image.height/2 - point[0, 1]))
        cv.PutText(image, name, point, font, cv.CV_RGB(255, 255, 0))

