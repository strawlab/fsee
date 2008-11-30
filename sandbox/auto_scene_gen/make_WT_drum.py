# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import fsee.scenegen.primlib as primlib
import os,math
import fsee.scenegen.osgwriter as osgwriter
import numpy

import VisionEgg
filename = os.path.join(
    VisionEgg.config.VISIONEGG_SYSTEM_DIR,
    "data","panorama.jpg")

# units in m
radius = 0.1
height = 0.3

z0 = 0
z1 = height

if 1:
    res = 72
    angles = numpy.linspace( 0.0, 360.0, res+1 )
    starts = angles[:-1]
    stops = angles[1:]

    D2R = math.pi/180.0
    start_x = radius*numpy.cos(starts*D2R)
    start_y = radius*numpy.sin(starts*D2R)
    stop_x = radius*numpy.cos(stops*D2R)
    stop_y = radius*numpy.sin(stops*D2R)
    geode = osgwriter.Geode(states=['GL_LIGHTING OFF'])

    floor = primlib.Prim()
    floor.texture_fname = os.path.split(filename)[1]
    count = 0
    quads = []

    for i in range(len(start_x)):
        start_frac = i/float(res)
        stop_frac = (i+1)/float(res)
        x1 = start_x[i]; y1 = start_y[i]
        x2 = stop_x[i]; y2 = stop_y[i]
        normal = (0,0,1) # wrong for now...
        floor.verts.append( [x1, y1, z0] )
        floor.verts.append( [x1, y1, z1] )
        floor.verts.append( [x2, y2, z1] )
        floor.verts.append( [x2, y2, z0] )
        floor.normals.append( normal )
        floor.normals.append( normal )
        floor.normals.append( normal )
        floor.normals.append( normal )

        floor.tex_coords.append( [start_frac,0] )
        floor.tex_coords.append( [start_frac,1] )
        floor.tex_coords.append( [stop_frac,1] )
        floor.tex_coords.append( [stop_frac,0] )
        quads.append( [count, count+1, count+2, count+3] )
        count += 4

    floor.prim_sets = [primlib.Quads( quads )]

    geode.append(floor.get_as_osg_geometry())

m = osgwriter.MatrixTransform(numpy.eye(4))
m.append(geode)

g = osgwriter.Group()
g.append(m)

fd = open('drum.osg','wb')
g.save(fd)
fd.close()

