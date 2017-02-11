import logging

from numpy import *

import stardb
import util

LOG = logging.getLogger(__name__)


__all__ = (
    'BarrelDistortionCameraModel',
    'CameraModel',
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


def _rotation_jacobians(points):
    """Return a Nx3x3 array of N jacobians of a single point under rotation.

    `points`: Nx3 array of points being rotated.

    Returns: N 3x3 Jacobians, columns corresponding with rotations clockwise
        about X, Y and Z axes respectively.  Rows corresponding with the
        point's X, Y and Z coordinates respectively. 

    """
    return (cross(identity(3)[newaxis, :, :],
                  points[:, :, newaxis],
                  axis=1))


def _projection_jacobians(points):
    """Return a Nx2x3 array of N jacobians of a single point under projection. 

    The projection is onto the Z=1 plane.

    """
    xy_columns = identity(2)[newaxis, :, :] / points[:, 2, newaxis, newaxis]

    z_column = -points[:, :2, newaxis] / points[:, 2, newaxis, newaxis] ** 2 

    return concatenate([xy_columns, z_column], axis=2)


def _numerical_full_jacobian(world_points, cam_matrix, pixel_scale, epsilon):
    """Compute the Jacobian of `_full_jacobian` numerically.

    The result should converge to `_full_jacobian` as epsilon tends to zero.

    """
    def f(v, x_rot=0., y_rot=0., z_rot=0., pixel_scale_inc=0.):
        cam_matrix_rotated = (matrix(cam_matrix) *
                                util.matrix_rotate_x(x_rot)[:3, :3] *
                                util.matrix_rotate_y(y_rot)[:3, :3] *
                                util.matrix_rotate_z(z_rot)[:3, :3])

        cam_model = BarrelDistortionCameraModel(pixel_scale + pixel_scale_inc,
                                                0., 600, 600)

        p = cam_model.world_vec_to_pixel(matrix(v).T, cam_matrix_rotated)
        return array(p)[:, newaxis]

    def d(v, x_rot=0., y_rot=0., z_rot=0., pixel_scale_inc=0.):
        return (f(v, x_rot, y_rot, z_rot, pixel_scale_inc) - f(v)) / epsilon

    return concatenate([concatenate([d(v, x_rot=epsilon),
                                     d(v, y_rot=epsilon),
                                     d(v, z_rot=epsilon),
                                     d(v, pixel_scale_inc=epsilon)],
                                    axis=1)
                        for v in world_points],
                       axis=0)


def _full_jacobian(world_points, cam_matrix, pixel_scale):
    """Return the Jacobian of `CameraModel.world_vec_to_pixel`.

    Columns correspond with:
     * Rotation of the camera matrix clockwise about the X-axis.
     * Rotation of the camera matrix clockwise about the Y-axis.
     * Rotation of the camera matrix clockwise about the Z-axis.
     * Increasing the pixel scale.

    Rows correspond with flattened pixel coordinates.

    """
    cam_matrix = array(cam_matrix)
    world_points = array(world_points)

    cam_points = matmul(world_points, cam_matrix)
    projected_points = cam_points[:, :2] / cam_points[:, newaxis, 2]

    # Get the Jacobians of each point under rotation. The rotation is negated
    # as the rotation is of the camera matrix rather than the points
    # themselves.
    prescale_Js = matmul(_projection_jacobians(cam_points),
                         -_rotation_jacobians(cam_points))

    Js = concatenate([prescale_Js * pixel_scale,
                      projected_points[:, :, newaxis]],
                     axis=2) * array([1, -1])[newaxis, :, newaxis]

    return concatenate(Js, axis=0)


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

    @staticmethod
    def _params_from_correspondences_approx(vecs,
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
        cam_matrix = matmul(U, Vt)
        if linalg.det(cam_matrix) < 0:
            cam_matrix = matmul(U * [1, 1, -1], Vt)

        return pixel_scale, cam_matrix

    @classmethod
    def make_from_correspondences(cls,
                                  vecs,
                                  im_coords,
                                  im_width,
                                  im_height,
                                  num_iterations):
        vecs = vstack([array(v).T for v in vecs])
        im_coords = array(list(im_coords))

        pixel_scale, cam_matrix = cls._params_from_correspondences_approx(
                                    vecs, im_coords, im_width, im_height)
    
        for iteration_index in range(num_iterations):
            cam = cls(pixel_scale, 0., im_width, im_height)

            J = _full_jacobian(vecs, cam_matrix, pixel_scale)

            err = im_coords - stack([array(cam.world_vec_to_pixel(
                                               matrix(v).T,
                                               cam_matrix))
                                      for v in vecs])
            LOG.debug("Iteration %s, error %s",
                      iteration_index, 
                      linalg.norm(err))

            x_rot, y_rot, z_rot, pixel_scale_increase = (
                    matmul(linalg.pinv(J), err.reshape((-1, 1)))).flat

            pixel_scale += pixel_scale_increase
            cam_matrix = matmul(cam_matrix,
                                array(util.matrix_rotate_x(x_rot)[:3, :3] *
                                      util.matrix_rotate_y(y_rot)[:3, :3] *
                                      util.matrix_rotate_z(z_rot)[:3, :3]))

        cam = cls(pixel_scale, 0., im_width, im_height)

        return cam, cam_matrix

