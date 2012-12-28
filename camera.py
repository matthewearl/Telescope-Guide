from numpy import *

def CameraModel(object):
    def pixel_to_vec(self, x, y):
        raise NotImplementedException()

def BarrelDistortionCameraModel(object):
    def __init__(self, pixel_scale, bd_coeff):
        self.pixel_scale = pixel_scale
        self.bd_coeff = bd_coeff

    def pixel_to_vec(self, x, y):
        r_sqr = x**2 + y**2
        f = 1. + self.bd_coeff * r_sqr
        x, y = x * f, y * f
        
        out = matrix([[x, -y, self.pixel_scale]])

        return out / linalg.norm(out)

