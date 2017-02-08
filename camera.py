from numpy import *

import stardb


__all__ = (
    'CameraModel',
    'BarrelDistortionCameraModel',
)


class CameraModel(object):
    def pixel_to_vec(self, x, y):
        raise NotImplementedException

    def pixel_to_world_vec(self, x, y, cam_matrix):
        return cam_matrix * self.pixel_to_vec(x, y)

    def world_vec_to_pixel(self, world_vec, cam_matrix):
        return self.vec_to_pixel(cam_matrix.T * world_vec)

    def generate_image_stars(self, star_db, cam_matrix):
        corners = cam_matrix * hstack(self.pixel_to_vec(x, y)
                                            for x in (0, self.image_width)
                                            for y in (0, self.image_height))
        centre = mean(corners, axis=1)
        radius = amax(linalg.norm(corners - centre, axis=0))

        for star, _ in star_db.search_vec(centre, radius):
            x, y = self.world_vec_to_pixel(star.vec, cam_matrix)
            if (0 <= x <= self.image_width and
                0 <= y <= self.image_height):
               yield stardb.ImageStar("{} (I)".format(star.id),
                                      (x, y),
                                      -star.mag)


class BarrelDistortionCameraModel(CameraModel):
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

    @classmethod
    def make_from_correspondences_approx(cls,
                                         vecs,
                                         im_coords,
                                         im_width,
                                         im_height):
        vecs = vstack([array(v).T for v in vecs])
        im_coords = array(list(im_coords))

        # Estimate the pixel scale based on mean deviation from the mean.
        pixel_scale = (std(im_coords - mean(im_coords, axis=0)) /
                            std(vecs - mean(vecs, axis=0)))

        # Rescale the image coordinates and place them on the plane Z=1. Flip
        # the Y-axis so that the parity matches that of `vecs`.
        plane_im_coords = ((im_coords - 0.5 * array([im_width, im_height])) *
                                array([1, -1]) / pixel_scale)
        plane_im_coords = concatenate([plane_im_coords,
                                       ones((im_coords.shape[0], 1))],
                                       axis=1)

        # Compute the camera matrix by finding the rotation that maps
        # `plane_im_coords` onto `vecs`. Do this using the Kabsch Algorithm:
        #     https://en.wikipedia.org/wiki/Kabsch_algorithm
        U, _, Vt = linalg.svd(matmul(vecs.T, plane_im_coords))
        R = matmul(U, Vt)
        if linalg.det(R) < 0:
            R = matmul(U * [1, 1, -1], Vt)

        return cls(pixel_scale, 0, im_width, im_height), R

