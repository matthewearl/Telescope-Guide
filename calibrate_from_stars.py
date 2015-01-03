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

    # @@@ Make these parameters configurable by command line, possibly with a
    # config file.
    im = cv.LoadImage('/media/vbox_d_drive/Photos/IMG_1559.JPG', True)

    star_to_pixel = {
            "HIP 6686": (65, 1705),
            "HIP 4427": (278, 1576),
            "HIP 3179": (533, 1730),
            "HIP 746": (656, 1456),
            "HIP 111944": (1829, 1449),
            "HIP 111104": (1937, 1396),
            "HIP 112917": (1803, 1577),
            "HIP 112242": (1919, 1559),
        }

    # The above coordinates were obtained from the non-RAW JPEG image. We wish
    # to obtain a calibration for the RAW CR2 image, so increase the pixel
    # coords to account for the pixels that CR2->JPG conversion chops off.
    star_to_pixel = dict((k, (v[0] + 11, v[1] + 5))
                            for k, v in star_to_pixel.iteritems())
        
    print calibrate(star_to_pixel,
                    (3906, 2602),    # RAW resolution.
                    annotate_image=im)

    cv.SaveImage("out.png", im)

