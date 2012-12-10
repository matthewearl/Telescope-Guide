#!/usr/bin/python

import math
import operator

margin=10
width=210
height=297
padding = 5
radius = 15.

__all__ = ['get_targets']

def gen_target(xpos, ypos, radius, number, num_bits=6):
    out = ""

    code = [[True, False] if number & 1<<bit else [False, True] for bit in reversed(xrange(num_bits))]
    code = reduce(operator.add, code)
    code = [False, True, True, True, True, False] + code 

    # Draw circles from the outside in
    for i in reversed(range(1,5)):
        params = {'radius': (1./6) * radius * i,
                  'fill': "black" if i % 2 == 0 else "white",
                  'x': xpos,
                  'y': ypos}
        out += "<circle r='{radius}' cx='{x}' cy='{y}' stroke='none' fill='{fill}' />\n".format(**params)

    # Draw the outer ring
    for idx, val in enumerate(code):
        start_angle = idx * 2. * math.pi / len(code)
        end_angle = (idx + 1) * 2 * math.pi / len(code)
        inner_radius = radius * 5./6 
        outer_radius = radius * 6./6 

        points = [(xpos + inner_radius * math.sin(start_angle), ypos - inner_radius * math.cos(start_angle)),
                  (xpos + inner_radius * math.sin(end_angle), ypos - inner_radius * math.cos(end_angle)),
                  (xpos + outer_radius * math.sin(end_angle), ypos - outer_radius * math.cos(end_angle)),
                  (xpos + outer_radius * math.sin(start_angle), ypos - outer_radius * math.cos(start_angle))]
     
        out += "<path d='M%f,%f A%f,%f 0 0,1 %f,%f L%f,%f A%f,%f 0 0,0 %f,%f Z'\n" \
               "      stroke='none' fill='%s' />\n" \
                % (points[0][0], points[0][1], # M 
                   inner_radius, inner_radius, # A radius
                   points[1][0], points[1][1], # A end
                   points[2][0], points[2][1], # L end
                   outer_radius, outer_radius, # A radius
                   points[3][0], points[3][1], # A end
                   'black' if val else 'white')

    # Draw a label
    size = 0.15 * radius
    out += "<text x='{x}' y='{y}' font-family='Verdana' font-size='{size}' fill='black'>{number}</text>n".format(
                x=(xpos-radius), y=(size + ypos-radius), size=size, number=number)

    return out 

def get_targets(roll_radius=None):
    out = {}
    y=margin
    n=0
    while y + 2 * radius < height - margin:
        x=margin
        while x + 2 * radius < width - margin:
            out[n] = (x + radius, height - (y + radius), 0.0)
            x += 2 * radius + padding
            n += 1
        y += 2 * radius + padding

    if roll_radius != None:
        out = dict((key, (roll_radius * math.sin(x/roll_radius), y, -roll_radius * math.cos(x/roll_radius))) for (key, (x, y, z)) in out.iteritems())

    return out

if __name__ == "__main__":
    print "<svg xmlns='http://www.w3.org/2000/svg' version='1.1'" \
          "     width='{width}mm' height='{height}mm' viewBox='0 0 {width} {height}'>".format(
                  width=width, height=height)
    for n, (x, y, z) in get_targets().iteritems():
        print gen_target(x, height - y, radius, n)

    print "</svg>"

