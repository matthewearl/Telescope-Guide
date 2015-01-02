#!/usr/bin/python

import numpy

import fit_lm
import stardb

__all__ = (
    'calibrate',
)

def calibrate(star_to_pixel, image_size, annotate_image=None):
    print "Loading database..."
    db = stardb.StarDatabase(stardb.hip_star_gen('data/hip_main.dat'))

    world_points = dict(
        (star_name, tuple(10000. * numpy.array(db[star_name].vec).flatten()))
            for star_name in star_to_pixel.keys())

    image_points = dict((name, (pos[0] - (image_size[0] / 2),
                                (image_size[1] / 2) - pos[1]))
                            for name, pos in star_to_pixel.iteritems())
    
    print "Solving"
    return fit_lm.solve(world_points, [image_points],
                 annotate_images=[annotate_image],
                 initial_matrices=[numpy.identity(4)],
                 change_ps=True, change_bd=True, change_pos=False)

if __name__ == "__main__":
    import cv

    im = cv.LoadImage('/media/vbox_d_drive/Photos/IMG_1559.JPG', True)
        
    print calibrate({
            "HIP 6686": (65, 1705),
            "HIP 4427": (278, 1576),
            "HIP 3179": (533, 1730),
            "HIP 746": (656, 1456),
            "HIP 111944": (1829, 1449),
            "HIP 111104": (1937, 1396),
            "HIP 112917": (1803, 1577),
            "HIP 112242": (1919, 1559),
        },
        (3888, 2592),
        annotate_image=im)

    cv.SaveImage("out.png", im)

