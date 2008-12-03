# Copyright (C) 2005-2008 California Institute of Technology, All rights reserved
# Author: Andrew D. Straw
import fsee.scenegen.primlib as primlib
import math
import fsee.scenegen.osgwriter as osgwriter
import numpy


# units in m
radius = 1.0
height = 0.8

z0 = 0
z1 = height

def make_floor( x0y0, x1y1, xtilesize=1, ytilesize=1, z=0 ):
    xo, yo = x0y0
    x1,y1 = x1y1
    xlen = x1-xo
    ylen = y1-yo
    xg = xtilesize
    yg = ytilesize
    maxi = int(math.ceil(xlen/xg))
    maxj = int(math.ceil(ylen/yg))
    floor = primlib.Prim()
    floor.texture_fname = "nearblack.png"
    count = 0
    quads = []
    for i in range(maxi):
        for j in range(maxj):
            floor.verts.append( [xo+i*xg,     yo+j*yg,     z] )
            floor.verts.append( [xo+i*xg,     yo+(j+1)*yg, z] )
            floor.verts.append( [xo+(i+1)*xg, yo+(j+1)*yg, z] )
            floor.verts.append( [xo+(i+1)*xg, yo+j*yg,     z] )
            floor.normals.append( (0,0,1) )
            floor.normals.append( (0,0,1) )
            floor.normals.append( (0,0,1) )
            floor.normals.append( (0,0,1) )
            floor.tex_coords.append( [0,0] )
            floor.tex_coords.append( [0,1] )
            floor.tex_coords.append( [1,1] )
            floor.tex_coords.append( [1,0] )
            quads.append( [count, count+1, count+2, count+3] )
            count += 4
    floor.prim_sets = [primlib.Quads( quads )]
    return floor

floor = make_floor( (-2.0, -2.0),(2.0,2.0), z=z0 )
ceil  = make_floor( (-2.0, -2.0),(2.0,2.0), z=z1 )

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

    for i in range(len(start_x)):
        x1 = start_x[i]; y1 = start_y[i]
        x2 = stop_x[i]; y2 = stop_y[i]

        wall = primlib.XZRect()
        if i%2==0:
            wall.texture_fname = 'redgreen.png'
        else:
            wall.texture_fname = 'greenred.png'
        wall.mag_filter = "NEAREST"
        z0 = 0
        z1 = height
        wall.verts = [[ x1, y1, z0],
                      [ x1, y1, z1],
                      [ x2, y2, z1],
                      [ x2, y2, z0]]
        geode.append(wall.get_as_osg_geometry())
    geode.append(floor.get_as_osg_geometry())
    geode.append(ceil.get_as_osg_geometry())

m = osgwriter.MatrixTransform(numpy.eye(4))
m.append(geode)

g = osgwriter.Group()
g.append(m)

fd = open('mamarama_checkerboard.osg','wb')
g.save(fd)
fd.close()

