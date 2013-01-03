from numpy import *

__all__ = ['CameraModel', 'BarrelDistortionCameraModel']

class CameraModel(object):
    def pixel_to_vec(self, x, y):
        raise NotImplementedException()

class BarrelDistortionCameraModel(object):
    def __init__(self, pixel_scale, bd_coeff, image_width, image_height):
        self.pixel_scale = pixel_scale
        self.bd_coeff = bd_coeff
        self.image_width = image_width
        self.image_height = image_height

    def pixel_to_vec(self, x, y):
        x = x - 0.5 * self.image_width
        y = - (y - 0.5 * self.image_height)

        r_sqr = x**2 + y**2
        f = 1. + self.bd_coeff * r_sqr
        x, y = x * f, y * f
        
        out = matrix([[x, y, self.pixel_scale]]).T

        return out / linalg.norm(out)

    def vec_to_pixel(self, vec):
        vec = self.pixel_scale * vec / vec[2, 0]
        sx, sy = vec[0, 0], vec[1, 0]

        ru = math.sqrt(sx**2 + sy**2)
        x = ru
        for i in xrange(20):
            x = x - (x * (1. + self.bd_coeff * x**2) - ru) / (1. + 3 * self.bd_coeff * x**2)
        rd = x

        px = sx * rd / ru
        py = sy * rd / ru

        px = px + 0.5 * self.image_width
        py = -py + 0.5 * self.image_height

        return px, py
