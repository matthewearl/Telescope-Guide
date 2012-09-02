from numpy import *

circles = [("%d%s" % (x + 8 * y, l), (25.0 * x, 85.0 * y + (0 if l == 'a' else 50.0))) for y in range(3) for x in range(8) for l in ('a', 'b')]

# sub_jacobian_point_<transformation>()
#
# Gives:
#    [[dPx/dtheta], [dPy/dtheta]]
#
# where Px,Py are the point projection functions:
#    Px(x, y, z) = x / z
#    Py(x, y, z) = y / z
# 
# and theta is the parameter of <transformation>.
# Transformations are applied to the camera, and rotations are CCW about the
# axis of rotation.

def sub_jacobian_point_translation_x(x, y, z):
    return matrix([[-1/z], [0.0]])

def sub_jacobian_point_translation_y(x, y, z):
    return matrix([[0.0], [-1/z]])

def sub_jacobian_point_translation_z(x, y, z):
    return matrix([[x/z**2], [y/z**2]])

def sub_jacobian_point_rotation_x(x, y, z):
    return matrix([[x*y/z**2], [1 + (y/z)**2]])

def sub_jacobian_point_rotation_y(x, y, z):
    return matrix([[1 + (x/z)**2], [x*y/z**2]])

def sub_jacobian_point_rotation_z(x, y, z):
    return matrix([[y/z], [-x/z]])

def sub_jacobian_point(x, y, z):
    fns = [sub_jacobian_point_translation_x,
           sub_jacobian_point_translation_y,
           sub_jacobian_point_translation_z,
           sub_jacobian_point_rotation_x,
           sub_jacobian_point_rotation_y,
           sub_jacobian_point_rotation_z]

    return hstack(fn(x, y, z) for fn in fns)

def make_jacobian(points):
    return vstack(sub_jacobian_point(*p) for p in points)
